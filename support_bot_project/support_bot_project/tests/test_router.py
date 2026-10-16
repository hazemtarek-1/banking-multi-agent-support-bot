# AI-GENERATED SECTION START
from src.router import INTENT_TO_AGENT


def test_all_intents_have_agents():
    assert set(INTENT_TO_AGENT) == {
        "card_payment_fee_charged",
        "cash_withdrawal_charge",
        "card_not_working",
        "order_physical_card",
        "country_support",
    }
    assert set(INTENT_TO_AGENT.values()) == {"billing", "technical", "booking", "general"}
# AI-GENERATED SECTION END
