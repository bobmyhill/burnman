from create_dataset import create_dataset

from datasets import Carlson_Lindsley_1988_CMS_opx_cpx
from datasets import endmember_reactions
from datasets import Frost_2003_FMASO_garnet
from datasets import Frost_2003_CFMASO_garnet
from datasets import Frost_2003_fper_ol_wad_rw
from datasets import ONeill_1987_QFI
from datasets import ONeill_1987_QFM
from datasets import Perkins_et_al_1981_MAS_py_opx
from datasets import Tsujino_et_al_2019_FMS_wad_rw
from datasets import Woodland_ONeill_1993_FASO_alm_sk


def compile_assemblages():
    dataset, storage, labels = create_dataset(import_assemblages=False)

    assemblages = []
    for xpt_set in [Carlson_Lindsley_1988_CMS_opx_cpx,
                    endmember_reactions,
                    Frost_2003_FMASO_garnet,
                    Frost_2003_CFMASO_garnet,
                    Frost_2003_fper_ol_wad_rw,
                    ONeill_1987_QFI,
                    ONeill_1987_QFM,
                    Perkins_et_al_1981_MAS_py_opx,
                    Tsujino_et_al_2019_FMS_wad_rw]:
        assemblages.extend(xpt_set.get_assemblages(dataset))

    # ass = Woodland_ONeill_1993_FASO_alm_sk.get_assemblages(dataset)
    return assemblages


assemblages = compile_assemblages()
for a in assemblages:
    print([b.name for b in a.phases])
    if hasattr(a, "stored_compositions"):
        for stored_composition in a.stored_compositions:
            if len(stored_composition) > 1:
                print(stored_composition[0])
                # print(stored_composition[1])
