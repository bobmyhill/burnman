from create_dataset import create_dataset

from datasets import Beyer_et_al_2021_NCFMASO
from datasets import Carlson_Lindsley_1988_CMS_opx_cpx
from datasets import endmember_reactions
from datasets import Frost_2003_FMASO_garnet
from datasets import Frost_2003_CFMASO_garnet
from datasets import Frost_2003_fper_ol_wad_rw
from datasets import Gasparik_1989_CMAS_px_gt
from datasets import Gasparik_1989_MAS_px_gt
from datasets import Gasparik_1989_NCMAS_px_gt
from datasets import Gasparik_1989_NMAS_px_gt
from datasets import Gasparik_1992_MAS_px_gt
from datasets import Gasparik_Newton_1984_MAS_opx_sp_fo
from datasets import Gasparik_Newton_1984_MAS_py_opx_sp_fo
from datasets import Hirose_et_al_2001_ilm_bdg_gt
from datasets import Jamieson_Roeder_1984_FMAS_ol_sp
from datasets import Katsura_et_al_2004_FMS_ol_wad
from datasets import Klemme_ONeill_2000_CMAS_opx_cpx_gt_ol_sp
from datasets import Liu_et_al_2016_gt_bdg_cor
from datasets import Liu_et_al_2017_bdg_cor
from datasets import Matsuzaka_et_al_2000_rw_wus_stv
from datasets import Nakajima_FR_2012_bdg_fper
from datasets import ONeill_1987_QFI
from datasets import ONeill_1987_QFM
from datasets import ONeill_Wood_1979_CFMAS_ol_gt
from datasets import ONeill_Wood_1979_ol_gt
from datasets import Perkins_et_al_1981_MAS_py_opx
from datasets import Perkins_Newton_1980_CMAS_opx_cpx_gt
from datasets import Perkins_Vielzeuf_1992_CFMS_ol_cpx
from datasets import Rohrbach_et_al_2007_NCFMASO_gt_cpx
from datasets import Seckendorff_ONeill_1992_ol_opx
from datasets import Tange_TNFS_2009_bdg_fper_stv
from datasets import Tsujino_et_al_2019_FMS_wad_rw
from datasets import Woodland_ONeill_1993_FASO_alm_sk


def compile_assemblages():
    dataset, storage, labels = create_dataset(import_assemblages=False)

    assemblages = []
    for xpt_set in [#Beyer_et_al_2021_NCFMASO,
                    #Carlson_Lindsley_1988_CMS_opx_cpx,
                    #endmember_reactions,
                    Frost_2003_FMASO_garnet,
                    Frost_2003_CFMASO_garnet,
                    Frost_2003_fper_ol_wad_rw,
                    Gasparik_1989_CMAS_px_gt,
                    Gasparik_1989_MAS_px_gt,
                    Gasparik_1989_NCMAS_px_gt,  # equilibrium state problem
                    Gasparik_1989_NMAS_px_gt,
                    Gasparik_1992_MAS_px_gt,
                    Gasparik_Newton_1984_MAS_opx_sp_fo,
                    Gasparik_Newton_1984_MAS_py_opx_sp_fo,
                    Hirose_et_al_2001_ilm_bdg_gt,
                    Jamieson_Roeder_1984_FMAS_ol_sp,
                    Katsura_et_al_2004_FMS_ol_wad,
                    Klemme_ONeill_2000_CMAS_opx_cpx_gt_ol_sp,
                    Liu_et_al_2016_gt_bdg_cor,
                    Liu_et_al_2017_bdg_cor,
                    Nakajima_FR_2012_bdg_fper,
                    Matsuzaka_et_al_2000_rw_wus_stv,
                    ONeill_1987_QFI,
                    ONeill_1987_QFM,
                    ONeill_Wood_1979_CFMAS_ol_gt,
                    ONeill_Wood_1979_ol_gt,
                    Perkins_et_al_1981_MAS_py_opx,
                    Perkins_Newton_1980_CMAS_opx_cpx_gt,
                    Perkins_Vielzeuf_1992_CFMS_ol_cpx,
                    Rohrbach_et_al_2007_NCFMASO_gt_cpx,
                    Seckendorff_ONeill_1992_ol_opx,
                    Tange_TNFS_2009_bdg_fper_stv,
                    Tsujino_et_al_2019_FMS_wad_rw,
                    Woodland_ONeill_1993_FASO_alm_sk
                    ]:
        print(f"Importing experiments from {xpt_set.__dict__['__name__']}")
        assemblages.extend(xpt_set.get_assemblages(dataset))

    # ass = Woodland_ONeill_1993_FASO_alm_sk.get_assemblages(dataset)
    return assemblages


assemblages = compile_assemblages()
print(f'{len(assemblages)} assemblages loaded successfully.')
if False:
    for a in assemblages:
        print([b.name for b in a.phases])
        if hasattr(a, "stored_compositions"):
            for stored_composition in a.stored_compositions:
                if len(stored_composition) > 1:
                    print(stored_composition[0])
                    # print(stored_composition[1])
