import json
import os
import  copy

# --------------------------------
# Read config file (environment variable or default)
# --------------------------------



# Parameters with defaults
drmin = 0.01 
drmax = 0.2 
selection = 0.2 

# -----------------------------
# Your existing code
# -----------------------------

processList = {
    
    #xsecs need to be scaled by 280/989 ...for xsec of ee -> H ...

    # Semileptonic processes
    "wzp6_ee_Henueqq_ecm125":    {"fraction":1},
    "wzp6_ee_Hqqenue_ecm125":    {"fraction":1},

    "wzp6_ee_Hmunumuqq_ecm125":    {"fraction":1},
    "wzp6_ee_Hqqmunumu_ecm125":    {"fraction":1},

    "wzp6_ee_Htaunutauqq_ecm125":    {"fraction":1, },
    "wzp6_ee_Hqqtaunutau_ecm125":    {"fraction":1, },


    "wzp6_ee_taunutauqq_ecm125":{"fraction":1},
    "wzp6_ee_tautauqq_ecm125":{"fraction":1},

    "wzp6_ee_enueqq_ecm125":{"fraction":1},
    "wzp6_ee_eeqq_ecm125":{"fraction":1},

    "wzp6_ee_munumuqq_ecm125":{"fraction":1},
    "wzp6_ee_mumuqq_ecm125":{"fraction":1},

    # # # Fully leptonic Processes
    "wzp6_ee_Htautau_ecm125" :  {"fraction":1 },
    "wzp6_ee_Hllnunu_ecm125":   {"fraction":1 },

    "wzp6_ee_eenunu_ecm125":    {"fraction":1,},
    "wzp6_ee_mumununu_ecm125":  {"fraction":1,},
    "wzp6_ee_tautaununu_ecm125":{"fraction":1, },
    "wzp6_ee_l1l2nunu_ecm125":  {"fraction":1, },
    "wzp6_ee_tautau_ecm125" :   {"fraction":1},

    # # # Fully hadronic Processes
    "wzp6_ee_Hgg_ecm125":       {"fraction":1},
    "wzp6_ee_Hbb_ecm125" :      {"fraction":1},

    "wzp6_ee_qq_ecm125":        {"fraction":1},
    "p8_ee_ZZ_4tau_ecm125":     {"fraction":1},
    
}

outputDir   = "./output"
inputDir    = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA"
nCPUS       = -1
includePaths = ["../src/functions.h"]


model_name = "fccee_flavtagging_edm4hep_wc"

url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
url_preproc = "{}/{}.json".format(url_model_dir, model_name)
url_model = "{}/{}.onnx".format(url_model_dir, model_name)

model_dir = (
    "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_7classes_12_04_2023/"
)
local_preproc = "{}/{}.json".format(model_dir, model_name)
local_model = "{}/{}.onnx".format(model_dir, model_name)

def get_file_path(url, filename):
    if os.path.exists(filename):
        return os.path.abspath(filename)
    else:
        urllib.request.urlretrieve(url, os.path.basename(url))
        return os.path.basename(url)


weaver_preproc = get_file_path(url_preproc, local_preproc)
weaver_model = get_file_path(url_model, local_model)

from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from addons.FastJet.jetClusteringHelper import (
    ExclusiveJetClusteringHelper,
)

jetFlavourHelper = None
jetClusteringHelper = None

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
        df = df.Define("photons_all", "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)",)
        df = df.Define("photons", "FCCAnalyses::ReconstructedParticle::sel_p(20)(photons_all)",)

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

        df = df.Define("photons_iso", "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.2)(photons, ChargedHadrons)",)
        df = df.Define("photons_sel_iso","FCCAnalyses::ZHfunctions::sel_iso(0.2)(photons, photons_iso)",)

        # Electron variables:
        df = df.Define("IsoMuonNum", "muons_sel_iso.size()")
        df = df.Define("Iso_Electrons_No", "electrons_sel_iso.size()")



        df = df.Filter(" IsoMuonNum == 1 ")


        df = df.Define("IsoElectron_4p", "FCCAnalyses::ReconstructedParticle::get_tlv(electrons_sel_iso)",)
        df = df.Define("IsoMuon_4p", "FCCAnalyses::ReconstructedParticle::get_tlv(muons_sel_iso)",)


        df = df.Define("MissingE_4p", "FCCAnalyses::ReconstructedParticle::get_tlv(MissingET)")
        df = df.Define("Missing_Pt", "MissingE_4p[0].Pt()",) 
        df = df.Filter("Missing_Pt > 3  ")

        #df = df.Define("EnuM" , " (MissingE_4p[0]+IsoElectron_4p[0]).M()")
        #df = df.Define("MunuM" , " (MissingE_4p[0]+IsoMuon_4p[0]).M()")
# ______________________________________________________________________________________________________
    # create a new collection of reconstructed particles removing electrons with p>#

        df = df.Define("ReconstructedParticles_nophotons",
                        "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, photons)",)

        df = df.Define("ReconstructedParticlesNoElectrons",
                        "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles_nophotons,electrons)",)

        df = df.Define("ReconstructedParticlesNoleptonsNoPhotons",
                        "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticlesNoElectrons,muons)",)
 # ____________________________________________________________________________________________
 # 2. Jet Clustring using "jetClusteringHelper": (Durham-kt)
 ##########################################    
        global jetClusteringHelper
        global jetFlavourHelper

    # define jet and run clustering parameters
    # name of collections in EDM root files
        collections = {
            "GenParticles": "Particle",
            "PFParticles": "ReconstructedParticles",
            "PFTracks": "EFlowTrack",
            "PFPhotons": "EFlowPhoton",
            "PFNeutralHadrons": "EFlowNeutralHadron",
            "TrackState": "EFlowTrack_1",
            "TrackerHits": "TrackerHits",
            "CalorimeterHits": "CalorimeterHits",
            "dNdx": "EFlowTrack_2",
            "PathLength": "EFlowTrack_L",
            "Bz": "magFieldBz",
        }      

        collections_noleptons_nophotons = copy.deepcopy(collections)
        collections_noleptons_nophotons["PFParticles"] = "ReconstructedParticlesNoleptonsNoPhotons"

        jetClusteringHelper = ExclusiveJetClusteringHelper(collections_noleptons_nophotons["PFParticles"], 2) # for Njet=2
        df = jetClusteringHelper.define(df)
        
        ## define jet flavour tagging parameters

        jetFlavourHelper = JetFlavourHelper(
            collections_noleptons_nophotons,
            jetClusteringHelper.jets,
            jetClusteringHelper.constituents,
        )
        ## define observables for tagger
        df = jetFlavourHelper.define(df)

        ## tagger inference
        df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)

        df = df.Filter("event_njet > 1")

        df = df.Define("Jets_p4", "JetConstituentsUtils::compute_tlv_jets({})".format(jetClusteringHelper.jets),)

        df = df.Define("Jets_InMa",  "JetConstituentsUtils::InvariantMass(Jets_p4[0], Jets_p4[1])",)

        df = df.Filter("Jets_InMa < 52.85" )
        #df = df.Define("Jets_InMa < 100")


        return df

    def output():
        branchList = [
        ]
        return branchList
