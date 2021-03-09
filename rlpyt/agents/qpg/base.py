
from rlpyt.utils.collections import namedarraytuple

AgentInfo = namedarraytuple("AgentInfo", ["dist_info"])
AgentInfoRnn = namedarraytuple("AgentInfoRnn",
    ["dist_info", "prev_rnn_state"])
