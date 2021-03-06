
import psutil
import time
import torch
import math
from collections import deque
import numpy as np

from rlpyt.runners.base import BaseRunner
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter


class MinibatchRlBase(BaseRunner):  #* BaseRunner not implemented
	"""
	Implements startup, logging, and agent checkpointing functionality, to be
	called in the `train()` method of the subclassed runner.  Subclasses will
	modify/extend many of the methods here.

	Args:
		algo: The algorithm instance.
		agent: The learning agent instance.
		sampler: The sampler instance.
		n_steps (int): Total number of environment steps to run in training loop.
		seed (int): Random seed to use, if ``None`` will generate randomly.
		affinity (dict): Hardware component assignments for sampler and algorithm.
		log_interval_steps (int): Number of environment steps between logging to csv.
	"""

	_eval = False

	def __init__(
			self,
			algo,
			agent,
			sampler,
			n_steps,
			min_save_args=None, #!
			running_std_thres=40, #!
			running_window_size=3, #!
			seed=None,
			affinity=None,
			log_interval_steps=1e5,
			):
		n_steps = int(n_steps)
		log_interval_steps = int(log_interval_steps)
		affinity = dict() if affinity is None else affinity
		save__init__args(locals())
		self.min_itr_learn = getattr(self.algo, 'min_itr_learn', 0)

	def startup(self):
		"""
		Sets hardware affinities, initializes the following: 1) sampler (which
		should initialize the agent), 2) agent device and data-parallel wrapper (if applicable),
		3) algorithm, 4) logger.
		"""
		p = psutil.Process()
		try:
			if (self.affinity.get("master_cpus", None) is not None and
					self.affinity.get("set_affinity", True)):
				p.cpu_affinity(self.affinity["master_cpus"])
			cpu_affin = p.cpu_affinity()
		except AttributeError:
			cpu_affin = "UNAVAILABLE MacOS"
		# logger.log(f"Runner {getattr(self, 'rank', '')} master CPU affinity: "
			# f"{cpu_affin}.")
		if self.affinity.get("master_torch_threads", None) is not None:
			torch.set_num_threads(self.affinity["master_torch_threads"])
		# logger.log(f"Runner {getattr(self, 'rank', '')} master Torch threads: "
			# f"{torch.get_num_threads()}.")
		if self.seed is None:
			self.seed = make_seed()
		set_seed(self.seed)
		self.rank = rank = getattr(self, "rank", 0)
		self.world_size = world_size = getattr(self, "world_size", 1)
		examples = self.sampler.initialize(
			agent=self.agent,  # Agent gets initialized in sampler.
			affinity=self.affinity,
			seed=self.seed + 1,
			bootstrap_value=getattr(self.algo, "bootstrap_value", False),
			traj_info_kwargs=self.get_traj_info_kwargs(),
			rank=rank,
			world_size=world_size,
		)
		self.itr_batch_size = self.sampler.batch_spec.size * world_size
		n_itr = self.get_n_itr()
		self.agent.to_device(self.affinity.get("cuda_idx", None))
		if world_size > 1:
			self.agent.data_parallel()
		self.algo.initialize(
			agent=self.agent,
			n_itr=n_itr,
			batch_spec=self.sampler.batch_spec,
			mid_batch_reset=self.sampler.mid_batch_reset,
			examples=examples,
			world_size=world_size,
			rank=rank,
		)
		self.initialize_logging()
		return n_itr


	def get_traj_info_kwargs(self):
		"""
		Pre-defines any TrajInfo attributes needed from elsewhere e.g.
		algorithm discount factor.
		"""
		return dict(discount=getattr(self.algo, "discount", 1))

	def get_n_itr(self):
		"""
		Determine number of train loop iterations to run.  Converts logging
		interval units from environment steps to iterations.
		"""
		# Log at least as often as requested (round down itrs):
		log_interval_itrs = max(self.log_interval_steps //
			self.itr_batch_size, 1)
		n_itr = self.n_steps // self.itr_batch_size
		if n_itr % log_interval_itrs > 0:  # Keep going to next log itr.
			n_itr += log_interval_itrs - (n_itr % log_interval_itrs)
		self.log_interval_itrs = log_interval_itrs
		self.n_itr = n_itr
		# logger.log(f"Running {n_itr} iterations of minibatch RL.")
		return n_itr

	def initialize_logging(self):
		self._opt_infos = {k: list() for k in self.algo.opt_info_fields}
		self._start_time = self._last_time = time.time()
		self._cum_time = 0.
		self._cum_completed_trajs = 0
		self._last_update_counter = 0

	def shutdown(self):
		logger.log("Training complete.")
		# self.pbar.stop()
		self.sampler.shutdown()

	def get_itr_snapshot(self, itr):
		"""
		Returns all state needed for full checkpoint/snapshot of training run,
		including agent parameters and optimizer parameters.
		"""
		return dict(
			itr=itr,
			cum_steps=itr * self.sampler.batch_size * self.world_size,
			agent_state_dict=self.agent.state_dict(),
			optimizer_state_dict=self.algo.optim_state_dict(),
			# replay_buffer_dict=self.algo.replay_buffer_dict(),
		)

	def save_itr_snapshot(self, itr, save_cur):
		"""
		Calls the logger to save training checkpoint/snapshot (logger itself
		may or may not save, depending on mode selected).
		"""
		# logger.log("saving snapshot...")
		params = self.get_itr_snapshot(itr)
		logger.save_itr_params(itr, params, save_cur)
		# logger.log("saved")

	def store_diagnostics(self, itr, traj_infos, opt_info):
		"""
		Store any diagnostic information from a training iteration that should
		be kept for the next logging iteration.
		"""
		self._cum_completed_trajs += len(traj_infos)
		for k, v in self._opt_infos.items():
			new_v = getattr(opt_info, k, [])
			v.extend(new_v if isinstance(new_v, list) else [new_v])
		# self.pbar.update((itr + 1) % self.log_interval_itrs)

	def log_diagnostics(self, itr, traj_infos=None, eval_time=0, save_cur=False, prefix='Diagnostics/'):
		"""
		Write diagnostics (including stored ones) to csv via the logger.
		"""
		if itr >= self.min_itr_learn - 1:
			self.save_itr_snapshot(itr, save_cur)
		new_time = time.time()
		self._cum_time = new_time - self._start_time
		train_time_elapsed = new_time - self._last_time - eval_time
		new_updates = self.algo.update_counter - self._last_update_counter
		new_samples = (self.sampler.batch_size * self.world_size *
			self.log_interval_itrs)
		updates_per_second = (float('nan') if itr == 0 else
			new_updates / train_time_elapsed)
		samples_per_second = (float('nan') if itr == 0 else
			new_samples / train_time_elapsed)
		replay_ratio = (new_updates * self.algo.batch_size * self.world_size /
			new_samples)
		cum_replay_ratio = (self.algo.batch_size * self.algo.update_counter /
			((itr + 1) * self.sampler.batch_size))  # world_size cancels.
		cum_steps = (itr + 1) * self.sampler.batch_size * self.world_size

		with logger.tabular_prefix(prefix):
			if self._eval:
				logger.record_tabular('CumTrainTime',
					self._cum_time - self._cum_eval_time)  # Already added new eval_time.
			logger.record_tabular('Iteration', itr)
			logger.record_tabular('CumTime (s)', self._cum_time)
			logger.record_tabular('CumSteps', cum_steps)
			logger.record_tabular('CumCompletedTrajs', self._cum_completed_trajs)
			logger.record_tabular('CumUpdates', self.algo.update_counter)
			logger.record_tabular('StepsPerSecond', samples_per_second)
			logger.record_tabular('UpdatesPerSecond', updates_per_second)
			logger.record_tabular('ReplayRatio', replay_ratio)
			logger.record_tabular('CumReplayRatio', cum_replay_ratio)
		self._log_infos(traj_infos)
		logger.dump_tabular(with_prefix=False)

		self._last_time = new_time
		self._last_update_counter = self.algo.update_counter
		if itr < self.n_itr - 1:
			logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
			# self.pbar = ProgBarCounter(self.log_interval_itrs)

	def _log_infos(self, traj_infos=None):
		"""
		Writes trajectory info and optimizer info into csv via the logger.
		Resets stored optimizer info.
		"""
		if traj_infos is None:
			traj_infos = self._traj_infos
		if traj_infos:
			for k in traj_infos[0]:
				if (not k.startswith("_")) and (k != 'initial_state') and (k != 'img_path'):
					logger.record_tabular_misc_stat(k, [info[k] for info in traj_infos])

		if self._opt_infos:
			for k, v in self._opt_infos.items():
				logger.record_tabular_misc_stat(k, v)
		self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)


class MinibatchRl(MinibatchRlBase):
	"""
	Runs RL on minibatches; tracks performance online using learning
	trajectories.
	"""

	def __init__(self, log_traj_window=100, **kwargs):
		"""
		Args: 
			log_traj_window (int): How many trajectories to hold in deque for computing performance statistics.
		"""
		super().__init__(**kwargs)
		self.log_traj_window = int(log_traj_window)

	def train(self):
		"""
		Performs startup, then loops by alternating between
		``sampler.obtain_samples()`` and ``algo.optimize_agent()``, logging
		diagnostics at the specified interval.
		"""
		n_itr = self.startup()
		for itr in range(n_itr):
			logger.set_iteration(itr)
			with logger.prefix(f"itr #{itr} "):
				self.agent.sample_mode(itr)  # Might not be this agent sampling.
				samples, traj_infos = self.sampler.obtain_samples(itr)
				self.agent.train_mode(itr)
				opt_info = self.algo.optimize_agent(itr, samples)
				self.store_diagnostics(itr, traj_infos, opt_info)
				if (itr + 1) % self.log_interval_itrs == 0:
					self.log_diagnostics(itr)
		self.shutdown()

	def initialize_logging(self):
		self._traj_infos = deque(maxlen=self.log_traj_window)
		self._new_completed_trajs = 0
		logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
		super().initialize_logging()
		# self.pbar = ProgBarCounter(self.log_interval_itrs)

	def store_diagnostics(self, itr, traj_infos, opt_info):
		self._new_completed_trajs += len(traj_infos)
		self._traj_infos.extend(traj_infos)
		super().store_diagnostics(itr, traj_infos, opt_info)

	def log_diagnostics(self, itr, prefix='Diagnostics/'):
		with logger.tabular_prefix(prefix):
			logger.record_tabular('NewCompletedTrajs', self._new_completed_trajs)
			logger.record_tabular('StepsInTrajWindow',
				sum(info["Length"] for info in self._traj_infos))
		super().log_diagnostics(itr, prefix=prefix)
		self._new_completed_trajs = 0


class MinibatchRlEval(MinibatchRlBase):
	"""
	Runs RL on minibatches; tracks performance offline using evaluation
	trajectories.
	"""

	_eval = True

	def train(self, return_buffer=False, check_running=True):
		"""
		Performs startup, evaluates the initial agent, then loops by
		alternating between ``sampler.obtain_samples()`` and
		``algo.optimize_agent()``.  Pauses to evaluate the agent at the
		specified log interval.
		"""
		best_itr = 0
		# initial_pi_loss = None
		# initial_q_loss = None
		# min_save_itr = self.min_save_args['min_save_itr']
		# min_save_pi_loss_ratio = self.min_save_args['min_save_pi_loss_ratio']
		# min_save_q_loss_ratio = self.min_save_args['min_save_q_loss_ratio']
		eval_reward_avg_all = []

		n_itr = self.startup()

		# Evaluate first - initialize initial and best eval reward (before running random exploration) - load policy first, and then reset
		with logger.prefix(f"itr eval"):
			self.agent.load_state_dict()
			eval_traj_infos, eval_time = self.evaluate_agent(itr=0)
			ini_eval_reward_avg = self.get_eval_reward(eval_traj_infos)
			# self.log_diagnostics(0, eval_traj_infos, eval_time)
			self.agent.reset_model()
			print(f'Initial eval reward: {ini_eval_reward_avg}')
			logger.log(f'Initial eval reward: {ini_eval_reward_avg}')

			# Initialize best reward
			best_eval_reward_avg = ini_eval_reward_avg
			# best_eval_reward_avg = -1000	# dummy

		for itr in range(n_itr):
			logger.set_iteration(itr)

			with logger.prefix(f"itr #{itr} "):
				self.agent.sample_mode(itr)
				samples, traj_infos = self.sampler.obtain_samples(itr)
				self.agent.train_mode(itr)
				opt_info = self.algo.optimize_agent(itr, samples)
			
				save_cur = False
				# Find if in min_itr_learn (random exploration using bad policy)
				if len(opt_info.piLoss) == 0:
					min_itr_learn = True
				else:
					min_itr_learn = False
				self.store_diagnostics(itr, traj_infos, opt_info)

				# It is possible that save_cur never satisfied in all itrs, then do not update policy for this retrain
				if (itr + 1) % self.log_interval_itrs == 0:
					eval_traj_infos, eval_time = self.evaluate_agent(itr)
					eval_reward_avg = self.get_eval_reward(eval_traj_infos)
				
					# Do not save at initial itrs
					if not min_itr_learn:
						eval_reward_avg_all += [eval_reward_avg]
						eval_reward_window = eval_reward_avg_all[-self.running_window_size:]

						# Get running average
						if len(eval_reward_avg_all) >= self.running_window_size:
							running_avg = np.mean(eval_reward_window)
						else:
							running_avg = -1000	# dummy
						
						# Get running std
						s0 = sum(1 for a in eval_reward_window)
						s1 = sum(a for a in eval_reward_window)
						s2 = sum(a*a for a in eval_reward_window)
						running_std = np.sqrt((s0 * s2 - s1 * s1)/(s0 * (s0-1)))
		
						# Determine if saving current snapshot
						if check_running and (running_avg-ini_eval_reward_avg) > 0 and eval_reward_avg > best_eval_reward_avg and running_std < self.running_std_thres:
							best_eval_reward_avg = eval_reward_avg
							best_itr = itr
							save_cur = True

						elif not check_running and eval_reward_avg > best_eval_reward_avg:
							best_eval_reward_avg = eval_reward_avg
							best_itr = itr
							save_cur = True

					self.log_diagnostics(itr, eval_traj_infos, eval_time, save_cur)
					if (itr + 1) % 10 == 0:
						logger.log(f'Average eval reward: {eval_reward_avg}')
						print(f'Average eval reward at itr {itr}: {eval_reward_avg}')
		self.shutdown()

		if return_buffer:
			return best_itr, self.algo.replay_buffer_dict()
		else:
			return best_itr

		# Determine if saving the params
		# pi_loss = opt_info.piLoss    
		# q_loss = opt_info.qLoss
		# if min_save_pi_loss_ratio is not None and len(pi_loss) > 0:
		# 	pi_loss = np.mean(pi_loss)
		# 	q_loss = np.mean(q_loss)
		# 	if initial_pi_loss is None:
		# 		initial_pi_loss = pi_loss
		# 		initial_q_loss = q_loss
		# 	if itr > min_save_itr and pi_loss < initial_pi_loss*min_save_pi_loss_ratio and q_loss < initial_q_loss*min_save_q_loss_ratio:
		# 		save_cur = True 
		# else:
		# 	save_cur = True

	def get_eval_reward(self, traj_infos):
		"""
		This is for determining when to save snapshot
		"""
		for k in traj_infos[0]:
			if k == 'Return':   # 'Length', 'Return', 'NonzeroRewards', 'DiscountedReturn', '_cur_discount'
				all = [info[k] for info in traj_infos]
				return np.mean(all)

	def evaluate_agent(self, itr):
		"""
		Record offline evaluation of agent performance, by ``sampler.evaluate_agent()``.
		"""
		if itr >= self.min_itr_learn - 1 or itr == 0:
			self.agent.eval_mode(itr)  # Might be agent in sampler.
			eval_time = -time.time()
			traj_infos = self.sampler.evaluate_agent(itr)
			eval_time += time.time()
		else:
			traj_infos = []
			eval_time = 0.0
		return traj_infos, eval_time

	def initialize_logging(self):
		super().initialize_logging()
		self._cum_eval_time = 0

	def log_diagnostics(self, itr, eval_traj_infos, eval_time, save_cur=False, prefix='Diagnostics/'):
		if not eval_traj_infos:
			logger.log("WARNING: had no complete trajectories in eval.")
		steps_in_eval = sum([info["Length"] for info in eval_traj_infos])
		with logger.tabular_prefix(prefix):
			logger.record_tabular('StepsInEval', steps_in_eval)
			logger.record_tabular('TrajsInEval', len(eval_traj_infos))
			self._cum_eval_time += eval_time
			logger.record_tabular('CumEvalTime', self._cum_eval_time)
		super().log_diagnostics(itr, eval_traj_infos, eval_time, save_cur, prefix=prefix)

class MinibatchRlEvalOnly(MinibatchRlBase):
	"""
	Only evaluating
	"""

	_eval = True

	def eval(self):
		"""
		Performs startup, evaluates the initial agent, then loops by
		alternating between ``sampler.obtain_samples()`` and
		``algo.optimize_agent()``.  Pauses to evaluate the agent at the
		specified log interval.
		"""
		n_itr = self.startup()
		itr = 0
		# logger.set_iteration(itr)
		# with logger.prefix(f"itr #{itr} "):
		eval_traj_infos, eval_time = self.evaluate_agent(itr)
		# for info in eval_traj_infos:
		#     print(info)
		eval_reward_all = self.get_eval_reward(eval_traj_infos)
		# self.log_diagnostics(itr, eval_traj_infos, eval_time, save_cur=False)
		self.sampler.shutdown()
		# print('\n\nAvg reward: ', eval_reward_avg)
		return eval_reward_all
		
	def get_eval_reward(self, traj_infos):
		"""
		This is for determining when to save snapshot
		"""
		all = [info['Return'] for info in traj_infos] # 'Length', 'Return', 'NonzeroRewards', 'DiscountedReturn', '_cur_discount'
		return all

	def evaluate_agent(self, itr):
		"""
		Record offline evaluation of agent performance, by ``sampler.evaluate_agent()``.
		"""
		# logger.log("Evaluating agent...")
		self.agent.eval_mode(itr)  # Might be agent in sampler.
		eval_time = -time.time()
		traj_infos = self.sampler.evaluate_agent(itr)
		eval_time += time.time()

		# logger.log("Evaluation runs complete.")
		return traj_infos, eval_time

	# def initialize_logging(self):
	#     super().initialize_logging()
	#     self._cum_eval_time = 0

	def startup(self):
		"""
		Sets hardware affinities, initializes the following: 1) sampler (which
		should initialize the agent), 2) agent device and data-parallel wrapper (if applicable),
		3) algorithm, 4) logger.
		"""
		p = psutil.Process()
		try:
			if (self.affinity.get("master_cpus", None) is not None and
					self.affinity.get("set_affinity", True)):
				p.cpu_affinity(self.affinity["master_cpus"])
			cpu_affin = p.cpu_affinity()
		except AttributeError:
			cpu_affin = "UNAVAILABLE MacOS"
		# logger.log(f"Runner {getattr(self, 'rank', '')} master CPU affinity: "
			# f"{cpu_affin}.")
		if self.affinity.get("master_torch_threads", None) is not None:
			torch.set_num_threads(self.affinity["master_torch_threads"])
		# logger.log(f"Runner {getattr(self, 'rank', '')} master Torch threads: "
			# f"{torch.get_num_threads()}.")
		if self.seed is None:
			self.seed = make_seed()
		set_seed(self.seed)
		self.rank = rank = getattr(self, "rank", 0)
		self.world_size = world_size = getattr(self, "world_size", 1)
		examples = self.sampler.initialize(
			agent=self.agent,  # Agent gets initialized in sampler.
			affinity=self.affinity,
			seed=self.seed + 1,
			bootstrap_value=getattr(self.algo, "bootstrap_value", False),
			traj_info_kwargs=self.get_traj_info_kwargs(),
			rank=rank,
			world_size=world_size,
		)
		self.itr_batch_size = self.sampler.batch_spec.size * world_size
		n_itr = self.get_n_itr()
		self.agent.to_device(self.affinity.get("cuda_idx", None))
		if world_size > 1:
			self.agent.data_parallel()
		self.algo.initialize(
			agent=self.agent,
			n_itr=n_itr,
			batch_spec=self.sampler.batch_spec,
			mid_batch_reset=self.sampler.mid_batch_reset,
			examples=examples,
			world_size=world_size,
			rank=rank,
		)
		# self.initialize_logging()
		return n_itr

	# def log_diagnostics(self, itr, eval_traj_infos, eval_time, save_cur=False, prefix='Diagnostics/'):

	#     steps_in_eval = sum([info["Length"] for info in eval_traj_infos])
	#     with logger.tabular_prefix(prefix):
	#         logger.record_tabular('StepsInEval', steps_in_eval)
	#         logger.record_tabular('TrajsInEval', len(eval_traj_infos))
	#         self._cum_eval_time += eval_time
	#         logger.record_tabular('CumEvalTime', self._cum_eval_time)
	#     super().log_diagnostics(itr, eval_traj_infos, eval_time, save_cur, prefix=prefix)
