import json
import os

# --------------------------------
# Read config file (environment variable or default)
# --------------------------------
config_file = os.environ.get("CONFIG_FILE", "config.json")

if os.path.exists(config_file):
    with open(config_file) as f:
        cfg = json.load(f)
else:
    cfg = {}

# Parameters with defaults
drmin = cfg.get("drmin", 0.01)
drmax = cfg.get("drmax", 0.5)
selection = cfg.get("selection", 0.4)

# -----------------------------
# Your existing code
# -----------------------------

processList = {
    "wzp6_ee_Henueqq_ecm125": {"fraction":0.1},
    "wzp6_ee_qq_ecm125": {"fraction":0.1},
    # "wzp6_ee_Hqqenue_ecm125": {"fraction":1},

    # "wzp6_ee_Hmunumuqq_ecm125": {"fraction":0.1},
    # "wzp6_ee_Hqqmunumu_ecm125": {"fraction":1},

    # "wzp6_ee_Htaunutauqq_ecm125": {"fraction":1},
    # "wzp6_ee_Hqqtaunutau_ecm125": {"fraction":1},

}

outputDir   = "./output"
inputDir    = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA"
nCPUS       = -1
includePaths = ["../functions.h"]

jetFlavourHelper = None
jetClusteringHelper = None

# Mandatory: RDFanalysis class where the use defines the operations on the TTree
class RDFanalysis:
    # ______________________________________________________________________________________________________				
    def analysers(df):
        # Define aliases
        df = df.Alias("Particle0", "Particle#0.index")
        df = df.Alias("Particle1", "Particle#1.index")
        df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
        df = df.Alias("Electron0", "Electron#0.index")
        df = df.Alias("Muon0", "Muon#0.index")
        df = df.Alias("Photon0", "Photon#0.index")
        df = df.Alias("Jet2","Jet#2.index")  


        df = df.Define("ChargedHadrons", "ReconstructedParticle2MC::selRP_ChargedHadrons( MCRecoAssociations0,MCRecoAssociations1,ReconstructedParticles,Particle)")
        # Leptons
        df = df.Define("electrons_all", "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)")
        df = df.Define("muons_all", "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)")
        df = df.Define("electrons", "FCCAnalyses::ReconstructedParticle::sel_p(0)(electrons_all)")
        df = df.Define("muons", "FCCAnalyses::ReconstructedParticle::sel_p(0)(muons_all)")

        # Isolation using JSON-configurable parameters
        df = df.Define(
            "electrons_iso",
            f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(electrons, ChargedHadrons)" #ReconstructedParticles)"
        )
        df = df.Define(
            "electrons_sel_iso",
            f"FCCAnalyses::ZHfunctions::sel_iso({selection})(electrons, electrons_iso)"
        )

        df = df.Define(
            "muons_iso",
            f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(muons, ChargedHadrons)" #ReconstructedParticles)"
        )
        df = df.Define(
            "muons_sel_iso",
            f"FCCAnalyses::ZHfunctions::sel_iso({selection})(muons, muons_iso)"
        )

        # Electron variables:
        df = df.Define("IsoMuonNum", "muons_sel_iso.size()")
        df = df.Define("Iso_Electrons_No", "electrons_sel_iso.size()")

        df = df.Filter("Iso_Electrons_No == 1  ")
        #df = df.Filter("IsoMuonNum == 0")

        return df

    def output():
        branchList = []
        return branchList
