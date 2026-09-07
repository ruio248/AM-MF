from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent
from agents.meanflowql import MeanFlowQL_Agent
from agents.meanflowql_beta import MeanFlowQL_Agent_BETA
from agents.native_meanflow import NativeMeanFlowAgent
from agents.am_meanflow import AMMeanFlowAgent
from agents.am_meanflow_target_changed import AMMeanFlowTargetChangedAgent

agents = dict(
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
    meanflowql = MeanFlowQL_Agent,
    meanflowql_beta = MeanFlowQL_Agent_BETA,
    native_meanflow = NativeMeanFlowAgent,
    am_meanflow = AMMeanFlowAgent,
    am_meanflow_target_changed = AMMeanFlowTargetChangedAgent,
)
