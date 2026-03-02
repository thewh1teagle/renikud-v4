from renikud_onnx import G2P

g2p = G2P("model.onnx")
print(g2p.phonemize("שלום עולם"))
print(g2p.phonemize("חבר הכנסת מיכאל איתן"))
