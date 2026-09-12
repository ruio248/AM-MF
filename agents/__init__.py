from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent
from agents.meanflowql import MeanFlowQL_Agent
from agents.meanflowql_beta import MeanFlowQL_Agent_BETA
from agents.am_meanflow_note import AMMeanFlowNoteAgent

agents = dict(
    am_meanflow_note=AMMeanFlowNoteAgent,
    fql=FQLAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
    meanflowql = MeanFlowQL_Agent,
    meanflowql_beta = MeanFlowQL_Agent_BETA,
)
