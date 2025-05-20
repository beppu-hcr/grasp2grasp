from grippers.allegro import AllegroLayer
from grippers.shadow import ShadowLayer
from grippers.humanhand import HumanHandLayer

def grabHandLayer(hand_name, device='cpu'):
    if hand_name == "Allegro":
        return AllegroLayer(device)
    elif hand_name == "shadow_hand":
        return ShadowLayer(device)
    elif hand_name == "HumanHand":
        return HumanHandLayer(device)
    else:
        raise ValueError(f"Hand name {hand_name} not supported")
    