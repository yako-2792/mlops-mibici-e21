"""The three questions of the project, by key."""

from mibici_e21.modeling.base import Question
from mibici_e21.modeling.clusters import ClusterQuestion
from mibici_e21.modeling.demand import DemandQuestion
from mibici_e21.modeling.risk import RiskQuestion

QUESTIONS: dict[str, Question] = {
    question.key: question for question in (DemandQuestion(), RiskQuestion(), ClusterQuestion())
}
