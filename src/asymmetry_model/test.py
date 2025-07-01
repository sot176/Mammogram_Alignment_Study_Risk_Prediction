from mirai_localized_dif_head import  extract_mirai_backbone
import sys

sys.path.append('../AsymMirai_master/')
mirai = extract_mirai_backbone('mgh_mammo_MIRAI_Base_May20_2019.p')
mirai.requires_grad = False
print(mirai)