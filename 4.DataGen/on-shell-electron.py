import json
import os
import copy
import urllib.request

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

            # Semileptonic processes
            "wzp6_ee_Henueqq_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_Hqqenue_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_Hmunumuqq_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_Hqqmunumu_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_Htaunutauqq_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_Hqqtaunutau_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_taunutauqq_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_tautauqq_ecm125": {"fraction": 1,'chunks': 100},
            "wzp6_ee_enueqq_ecm125": {"fraction": 1,'chunks': 100},
            "wzp6_ee_eeqq_ecm125": {"fraction": 1,'chunks': 100},
            "wzp6_ee_munumuqq_ecm125": {"fraction": 1,'chunks': 100},
            "wzp6_ee_mumuqq_ecm125": {"fraction": 1,'chunks': 100},
            # Fully leptonic Processes
            #"wzp6_ee_Htautau_ecm125": {"fraction": 1},
            #"wzp6_ee_Hllnunu_ecm125": {"fraction": 1},
            "wzp6_ee_eenunu_ecm125": {"fraction": 1,'chunks': 100},
            "wzp6_ee_mumununu_ecm125": {"fraction": 1,'chunks': 100},
            "wzp6_ee_tautaununu_ecm125": {"fraction": 1,'chunks': 100},
            "wzp6_ee_l1l2nunu_ecm125": {"fraction": 1,'chunks': 100},
            "wzp6_ee_tautau_ecm125": {"fraction": 1,'chunks': 100},
            # Fully hadronic Processes
            #"wzp6_ee_Hgg_ecm125": {"fraction": 1},
            #"wzp6_ee_Hbb_ecm125": {"fraction": 1},
            "wzp6_ee_qq_ecm125": {"fraction": 1,'chunks': 100},
            "p8_ee_ZZ_4tau_ecm125": {"fraction": 1,'chunks': 100},
}

outputDir = "/eos/experiment/fcc/ee/analyses/case-studies/higgs/electron_yukawa/DataGenReduced-CHCut/on-shell-electron/"
inputDir = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA"
nCPUS = -1
includePaths = ["functions.h", "GEOFunctions.h", "MELAFunctions.h","SortJets.h" ]

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
    """Return local file path if exists else download from url and return basename."""
    if os.path.exists(filename):
        return os.path.abspath(filename)
    else:
        urllib.request.urlretrieve(url, os.path.basename(url))
        return os.path.basename(url)


weaver_preproc = get_file_path(url_preproc, local_preproc)
weaver_model = get_file_path(url_model, local_model)

from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from addons.FastJet.jetClusteringHelper import ExclusiveJetClusteringHelper

# helpers to be used inside analysers
jetFlavourHelper = None
jetClusteringHelper = None

class RDFanalysis:

    @staticmethod
    def analysers(df):
        '''
        Analysis graph for electron Yukawa coupling measurement.
        '''
        from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
        from addons.FastJet.jetClusteringHelper import ExclusiveJetClusteringHelper



        # ===========================
        # Aliases
        # ===========================
        df = df.Alias("Particle0", "Particle#0.index")
        df = df.Alias("Particle1", "Particle#1.index")
        df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
        df = df.Alias("Electron0", "Electron#0.index")
        df = df.Alias("Muon0", "Muon#0.index")
        df = df.Alias("Photon0", "Photon#0.index")
        df = df.Alias("Jet2", "Jet#2.index")

        # ===========================
        # Missing energy variables
        # ===========================
        df = df.Define("MissingQuantities_4Vec", "FCCAnalyses::ReconstructedParticle::get_tlv(MissingET)")
        df = df.Define("MissingQuantities_Pt", "MissingQuantities_4Vec[0].Pt()")
        df = df.Filter("MissingQuantities_Pt > 3")
        df = df.Define("MissingQuantities_P", "MissingQuantities_4Vec[0].P()")
        df = df.Define("MissingQuantities_E", "MissingQuantities_4Vec[0].E()")
        df = df.Define("MissingQuantities_M", "MissingQuantities_4Vec[0].M()")
        df = df.Define("MissingQuantities_Theta", "MissingQuantities_4Vec[0].Theta()")
        df = df.Define("MissingQuantities_Phi", "MissingQuantities_4Vec[0].Phi()")
        df = df.Define("MissingQuantities_CosTheta", "MissingQuantities_4Vec[0].CosTheta()")
        df = df.Define("MissingQuantities_CosPhi", "TMath::Cos(MissingQuantities_Phi)")

        # ===========================
        # Photons and charged hadrons
        # ===========================
        df = df.Define("Photons_All", "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)")
        df = df.Define("Photons_Selected", "FCCAnalyses::ReconstructedParticle::sel_p(20)(Photons_All)")
        df = df.Define(
            "ChargedHadrons",
            "ReconstructedParticle2MC::selRP_ChargedHadrons(MCRecoAssociations0,MCRecoAssociations1,ReconstructedParticles,Particle)",
        )

        # ===========================
        # Leptons
        # ===========================
        df = df.Define("Electrons_All", "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)")
        df = df.Define("Muons_All", "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)")
        df = df.Define("Electrons_PreSelection", "FCCAnalyses::ReconstructedParticle::sel_p(0)(Electrons_All)")
        df = df.Define("Muons_PreSelection", "FCCAnalyses::ReconstructedParticle::sel_p(0)(Muons_All)")

        # ===========================
        # Isolation
        # ===========================
        df = df.Define(
            "Electrons_IsolationValue",
            f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(Electrons_PreSelection, ChargedHadrons)",
        )
        df = df.Define(
            "Electrons_Isolated",
            f"FCCAnalyses::ZHfunctions::sel_iso({selection})(Electrons_PreSelection, Electrons_IsolationValue)",
        )
        df = df.Define(
            "Electrons_Isolated_rp",
            f"FCCAnalyses::ZHfunctions::sel_iso_rp({selection})(Electrons_PreSelection, Electrons_IsolationValue)",
        )

        df = df.Define(
            "Muons_IsolationValue",
            f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(Muons_PreSelection, ChargedHadrons)",
        )
        df = df.Define(
            "Muons_Isolated",
            f"FCCAnalyses::ZHfunctions::sel_iso({selection})(Muons_PreSelection, Muons_IsolationValue)",
        )

        df = df.Define(
            "Muons_Isolated_rp",
            f"FCCAnalyses::ZHfunctions::sel_iso_rp({selection})(Muons_PreSelection, Muons_IsolationValue)",
        )

        df = df.Define(
            "Photons_IsolationValue",
            f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(Photons_Selected, ChargedHadrons)"
        )
        df = df.Define(
            "Photons_Isolated",
            f"FCCAnalyses::ZHfunctions::sel_iso({selection})(Photons_Selected, Photons_IsolationValue)"
        )

        # ===========================
        # Lepton counting and selection
        # ===========================
        df = df.Define("N_IsolatedMuons", "Muons_Isolated.size()")
        df = df.Define("N_IsolatedElectrons", "Electrons_Isolated.size()")

        # Require exactly one isolated electron and no isolated muons
        df = df.Filter("N_IsolatedElectrons == 1")
        df = df.Filter("N_IsolatedMuons == 0")

        # ===========================
        # Isolated particle 4-vectors
        # ===========================
        df = df.Define("IsolatedElectron_4Vec", "FCCAnalyses::ReconstructedParticle::get_tlv(Electrons_Isolated)")
        df = df.Define("IsolatedMuon_4Vec", "FCCAnalyses::ReconstructedParticle::get_tlv(Muons_Isolated)")
        df = df.Define("IsolatedPhotons_4Vec", "FCCAnalyses::ReconstructedParticle::get_tlv(Photons_Isolated)")

        # ===========================
        # Isolated photon properties
        # ===========================
        df = df.Define("IsolatedPhoton_Phi", "IsolatedPhotons_4Vec[0].Phi()")
        df = df.Define("IsolatedPhoton_Theta", "IsolatedPhotons_4Vec[0].Theta()")
        df = df.Define("IsolatedPhoton_E", "IsolatedPhotons_4Vec[0].E()")
        df = df.Define("IsolatedPhoton_CosTheta", "IsolatedPhotons_4Vec[0].CosTheta()")
        df = df.Define("IsolatedPhoton_CosPhi", "TMath::Cos(IsolatedPhoton_Phi)")
        df = df.Define("N_IsolatedPhotons", "Photons_Isolated.size()")
        df = df.Define("IsolatedPhoton_P", "IsolatedPhotons_4Vec[0].P()")

        # ===========================
        # Isolated electron properties
        # ===========================
        df = df.Define("IsolatedElectron_3Vec", "IsolatedElectron_4Vec[0].Vect()")
        df = df.Define("IsolatedElectron_P", "IsolatedElectron_4Vec[0].P()")
        df = df.Define("IsolatedElectron_M", "IsolatedElectron_4Vec[0].M()")
        df = df.Define("IsolatedElectron_Phi", "IsolatedElectron_4Vec[0].Phi()")
        df = df.Define("IsolatedElectron_Theta", "IsolatedElectron_4Vec[0].Theta()")
        df = df.Define("IsolatedElectron_E", "IsolatedElectron_4Vec[0].E()")
        df = df.Define("IsolatedElectron_CosTheta", "IsolatedElectron_4Vec[0].CosTheta()")
        df = df.Define("IsolatedElectron_CosPhi", "TMath::Cos(IsolatedElectron_Phi)")
        df = df.Define("IsolatedElectron_Charge", "FCCAnalyses::ReconstructedParticle::get_charge(Electrons_Isolated)[0]")

        # ===========================
        # Create collections with particles removed
        # ===========================
        df = df.Define(
            "RecoParticles_NoNoElectrons",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, Electrons_Isolated_rp)",
        )
        df = df.Define(
            "RecoParticles_NoLeptons",
            "FCCAnalyses::ReconstructedParticle::remove(RecoParticles_NoNoElectrons, Muons_Isolated_rp)",
        )

        # ===========================
        # Jet clustering setup
        # ===========================
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

        collections_noisoleptons = copy.deepcopy(collections)
        collections_noisoleptons["PFParticles"] = "RecoParticles_NoLeptons"

        jetClusteringHelper = ExclusiveJetClusteringHelper(
            collections_noisoleptons["PFParticles"], 2
        )
        df = jetClusteringHelper.define(df)

        # ===========================
        # Jet flavour tagger
        # ===========================
        jetFlavourHelper = JetFlavourHelper(
            collections_noisoleptons,
            jetClusteringHelper.jets,
            jetClusteringHelper.constituents,
        )
        df = jetFlavourHelper.define(df)

        # ===========================
        # Jet filtering and definitions
        # ===========================
        df = df.Filter("event_njet > 1")
        df = df.Define("Jets_4Vec", 
                                 "JetConstituentsUtils::compute_tlv_jets({})".format(jetClusteringHelper.jets))
        df = df.Define("DiJet_InvariantMass", "JetConstituentsUtils::InvariantMass(Jets_4Vec[0], Jets_4Vec[1])")
        df = df.Filter("DiJet_InvariantMass < 52.85")
        df = df.Filter("DiJet_InvariantMass > 4")

        # ===========================
        # Tagger inference
        # ===========================
        df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)

        # ===========================
        # Additional jet clustering
        # ===========================
        jetClusteringHelper_N2 = ExclusiveJetClusteringHelper("RecoParticles_NoLeptons", 2, "N2")
        df = jetClusteringHelper_N2.define(df)

        df = df.Define("JetClustering_d23", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 2))")
        df = df.Define("JetClustering_d34", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 3))")
        df = df.Define("Jets_Charge", "JetConstituentsUtils::get_charge({})".format(jetClusteringHelper.constituents))


        df = df.Define("jetc_isMu",       f"JetConstituentsUtils::get_isMu({jetClusteringHelper.constituents})") 
        df = df.Define("jetc_isEl",       f"JetConstituentsUtils::get_isEl({jetClusteringHelper.constituents})") 
        
        df = df.Define("jetc_isCH",  f"JetConstituentsUtils::get_isChargedHad({jetClusteringHelper.constituents})") 
        df = df.Define("jetc_isGAMMA",     f"JetConstituentsUtils::get_isGamma({jetClusteringHelper.constituents})") 
        df = df.Define("jetc_isNH", f"JetConstituentsUtils::get_isNeutralHad({jetClusteringHelper.constituents})")

        df = df.Define(f"jetc_nmu",    f"JetConstituentsUtils::count_type(jetc_isMu)") 
        df = df.Define(f"jetc_nel",    f"JetConstituentsUtils::count_type(jetc_isEl)") 
        df = df.Define(f"jetc_nCH",    f"JetConstituentsUtils::count_type(jetc_isCH)") 
        df = df.Define(f"jetc_nGAMMA",    f"JetConstituentsUtils::count_type(jetc_isGAMMA)") 
        df = df.Define(f"jetc_nNH",    f"JetConstituentsUtils::count_type(jetc_isNH)") 

        df = df.Define("Jet1_NConstituents", "jet_nconst[0] - jetc_nmu[0] - jetc_nel[0]")
        df = df.Define("Jet2_NConstituents", "jet_nconst[1] - jetc_nmu[1] - jetc_nel[1]")

        df = df.Filter("Jet1_NConstituents > 2")
        df = df.Filter("Jet2_NConstituents > 2")
        
        df = df.Filter("jetc_nCH[0] > 0")
        df = df.Filter("jetc_nCH[1] > 0")
        # ===========================
        # Jet 1 kinematics
        # ===========================
        df = df.Define("Jet1_3Vec", "Jets_4Vec[0].Vect()")
        df = df.Define("Jet1_P", "Jets_4Vec[0].P()")
        df = df.Define("Jet1_Eta", "Jets_4Vec[0].Eta()")
        df = df.Define("Jet1_Phi", "Jets_4Vec[0].Phi()")
        df = df.Define("Jet1_M", "Jets_4Vec[0].M()")
        df = df.Define("Jet1_E", "Jets_4Vec[0].E()")
        df = df.Define("Jet1_Theta", "Jets_4Vec[0].Theta()")
        df = df.Define("Jet1_CosTheta", "Jets_4Vec[0].CosTheta()")
        df = df.Define("Jet1_CosPhi", "TMath::Cos(Jet1_Phi)")
        # ===========================
        # Jet 2 kinematics
        # ===========================
        df = df.Define("Jet2_3Vec", "Jets_4Vec[1].Vect()")
        df = df.Define("Jet2_P", "Jets_4Vec[1].P()")
        df = df.Define("Jet2_Eta", "Jets_4Vec[1].Eta()")
        df = df.Define("Jet2_Phi", "Jets_4Vec[1].Phi()")
        df = df.Define("Jet2_M", "Jets_4Vec[1].M()")
        df = df.Define("Jet2_E", "Jets_4Vec[1].E()")
        df = df.Define("Jet2_Theta", "Jets_4Vec[1].Theta()")
        df = df.Define("Jet2_CosTheta", "Jets_4Vec[1].CosTheta()")
        df = df.Define("Jet2_CosPhi", "TMath::Cos(Jet2_Phi)")
        # ===========================
        # Jet energy and charge
        # ===========================
        df = df.Define("Jets_MaxEnergy", "TMath::Max(Jets_4Vec[0].E(), Jets_4Vec[1].E())")
        df = df.Define("Jets_MinEnergy", "TMath::Min(Jets_4Vec[0].E(), Jets_4Vec[1].E())")
        df = df.Define("Jet1_Charge", "ROOT::VecOps::Sum(Jets_Charge[0])")
        df = df.Define("Jet2_Charge", "ROOT::VecOps::Sum(Jets_Charge[1])")
        # ===========================
        # PAIR SYSTEM DEFINITIONS
        # ===========================
        # Leptonic W Boson (Electron + Neutrino)
        df = df.Define("LeptonicW_4Vec", "IsolatedElectron_4Vec[0] + MissingQuantities_4Vec[0]")
        df = df.Define("LeptonicW_P", "LeptonicW_4Vec.P()")
        df = df.Define("LeptonicW_Phi", "LeptonicW_4Vec.Phi()")
        df = df.Define("LeptonicW_CosPhi", "TMath::Cos(LeptonicW_Phi)")
        df = df.Define("LeptonicW_Theta", "LeptonicW_4Vec.Theta()")
        df = df.Define("LeptonicW_CosTheta", "LeptonicW_4Vec.CosTheta()")
        df = df.Define("LeptonicW_InvariantMass", "LeptonicW_4Vec.M()")
        
        # Hadronic W Boson (Jet1 + Jet2)
        df = df.Define("HadronicW_4Vec", "Jets_4Vec[0] + Jets_4Vec[1]")
        df = df.Define("HadronicW_P", "HadronicW_4Vec.P()")
        df = df.Define("HadronicW_Phi", "HadronicW_4Vec.Phi()")
        df = df.Define("HadronicW_CosPhi", "TMath::Cos(HadronicW_Phi)")
        df = df.Define("HadronicW_Theta", "HadronicW_4Vec.Theta()")
        df = df.Define("HadronicW_CosTheta", "HadronicW_4Vec.CosTheta()")
        df = df.Define("HadronicW_InvariantMass", "HadronicW_4Vec.M()")
        
        # Angular separations between the two W bosons
        df = df.Define("WW_DeltaR", "LeptonicW_4Vec.DeltaR(HadronicW_4Vec)")
        df = df.Define("WW_DeltaPhi", "LeptonicW_4Vec.DeltaPhi(HadronicW_4Vec)")
        df = df.Define("WW_DeltaTheta", "LeptonicW_Theta - HadronicW_Theta")
        df = df.Define("WW_DeltaInvariantMass", "LeptonicW_InvariantMass - HadronicW_InvariantMass")
        df = df.Define("WW_Angle", "LeptonicW_4Vec.Angle(HadronicW_4Vec.Vect())")
        df = df.Define("WW_CosAngle", "TMath::Cos(WW_Angle)")

        # ===========================
        # Event-level invariant mass
        # ===========================
        df = df.Define("Event_InvariantMass", "(Jets_4Vec[0] + Jets_4Vec[1] + MissingQuantities_4Vec[0] + IsolatedElectron_4Vec[0]).M()")

        # ===========================
        # On-shell and off-shell mass variables
        # ===========================
        df = df.Define("System_OnShellMass", "TMath::Max(HadronicW_InvariantMass, LeptonicW_InvariantMass)")
        df = df.Define("System_OffShellMass", "TMath::Min(HadronicW_InvariantMass, LeptonicW_InvariantMass)")

        df = df.Define("System_MaxCosTheta", "TMath::Cos(TMath::Max(HadronicW_Theta, LeptonicW_Theta))")
        df = df.Define("System_MinCosTheta", "TMath::Cos(TMath::Min(HadronicW_Theta, LeptonicW_Theta))")

        df = df.Define("System_MaxCosPhi", "TMath::Cos(TMath::Max(HadronicW_Phi, LeptonicW_Phi))")
        df = df.Define("System_MinCosPhi", "TMath::Cos(TMath::Min(HadronicW_Phi, LeptonicW_Phi))")

        df = df.Define("System_MaxCosP", "TMath::Cos(TMath::Max(HadronicW_P, LeptonicW_P))")
        df = df.Define("System_MinCosP", "TMath::Cos(TMath::Min(HadronicW_P, LeptonicW_P))")
        # ===========================
        # MELA Variables (production and decay angles)
        # ===========================
        df = df.Define(
            "MELA_Angles",
            "FCCAnalyses::MELA::MELACalculator::mela(Jets_4Vec[0], Jets_4Vec[1], MissingQuantities_4Vec[0], IsolatedElectron_4Vec[0], IsolatedElectron_Charge, Jet1_Charge, Jet2_Charge)",
        )
        df = df.Define("MELA_Phi", "MELA_Angles.phi")
        df = df.Define("MELA_CosPhi", "MELA_Angles.cosPhi")
        df = df.Define("MELA_Phi1", "MELA_Angles.phi1")
        df = df.Define("MELA_CosPhi1", "MELA_Angles.cosPhi1")
        df = df.Define("MELA_PhiStar", "MELA_Angles.phiStar")
        df = df.Define("MELA_CosPhiStar", "MELA_Angles.cosPhiStar")
        df = df.Define("MELA_ThetaStar", "MELA_Angles.thetaStar")
        df = df.Define("MELA_CosThetaStar", "MELA_Angles.cosThetaStar")
        df = df.Define("MELA_Theta1", "MELA_Angles.theta1")
        df = df.Define("MELA_CosTheta1", "MELA_Angles.cosTheta1")
        df = df.Define("MELA_Theta2", "MELA_Angles.theta2")
        df = df.Define("MELA_CosTheta2", "MELA_Angles.cosTheta2")

        # ===========================
        # Event-shape variables
        # ===========================
        df = df.Define("EventShape_Planarity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculatePlanarity(Jet1_3Vec, Jet2_3Vec, IsolatedElectron_3Vec)")
        df = df.Define("EventShape_Aplanarity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAplanarity(Jet1_3Vec, Jet2_3Vec, IsolatedElectron_3Vec)")
        df = df.Define("EventShape_Sphericity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateSphericity(Jet1_3Vec, Jet2_3Vec, IsolatedElectron_3Vec)")
        df = df.Define("EventShape_Asphericity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAsphericity(Jet1_3Vec, Jet2_3Vec, IsolatedElectron_3Vec)")

        # ===========================
        # Jet flavor tagging scores - Products and Logits
        # ===========================
        epsilon = 1e-3
        
        df = df.Define("JetFlavor_GluonProduct", "recojet_isG[0] * recojet_isG[1]")
        df = df.Define("JetFlavor_GluonProduct_Logit", f"TMath::Log((JetFlavor_GluonProduct + {epsilon}) / (1.0 - JetFlavor_GluonProduct + {epsilon}))")
        
        df = df.Define("JetFlavor_UpQuarkProduct", "recojet_isU[0] * recojet_isU[1]")
        df = df.Define("JetFlavor_UpQuarkProduct_Logit", f"TMath::Log((JetFlavor_UpQuarkProduct + {epsilon}) / (1.0 - JetFlavor_UpQuarkProduct + {epsilon}))")
        
        df = df.Define("JetFlavor_DownQuarkProduct", "recojet_isD[0] * recojet_isD[1]")
        df = df.Define("JetFlavor_DownQuarkProduct_Logit", f"TMath::Log((JetFlavor_DownQuarkProduct + {epsilon}) / (1.0 - JetFlavor_DownQuarkProduct + {epsilon}))")
        
        df = df.Define("JetFlavor_StrangeQuarkProduct", "recojet_isS[0] * recojet_isS[1]")
        df = df.Define("JetFlavor_StrangeQuarkProduct_Logit", f"TMath::Log((JetFlavor_StrangeQuarkProduct + {epsilon}) / (1.0 - JetFlavor_StrangeQuarkProduct + {epsilon}))")
        
        df = df.Define("JetFlavor_CharmQuarkProduct", "recojet_isC[0] * recojet_isC[1]")
        df = df.Define("JetFlavor_CharmQuarkProduct_Logit", f"TMath::Log((JetFlavor_CharmQuarkProduct + {epsilon}) / (1.0 - JetFlavor_CharmQuarkProduct + {epsilon}))")
        
        df = df.Define("JetFlavor_BottomQuarkProduct", "recojet_isB[0] * recojet_isB[1]")
        df = df.Define("JetFlavor_BottomQuarkProduct_Logit", f"TMath::Log((JetFlavor_BottomQuarkProduct + {epsilon}) / (1.0 - JetFlavor_BottomQuarkProduct + {epsilon}))")
        
        df = df.Define("JetFlavor_TauProduct", "recojet_isTAU[0] * recojet_isTAU[1]")
        df = df.Define("JetFlavor_TauProduct_Logit", f"TMath::Log((JetFlavor_TauProduct + {epsilon}) / (1.0 - JetFlavor_TauProduct + {epsilon}))")
        
        df = df.Define("JetFlavor_UpDownQuarkProduct", "recojet_isU[0] * recojet_isD[1]")
        df = df.Define("JetFlavor_UpDownQuarkProduct_Logit", f"TMath::Log((JetFlavor_UpDownQuarkProduct + {epsilon}) / (1.0 - JetFlavor_UpDownQuarkProduct + {epsilon}))")
        
        df = df.Define("JetFlavor_DownUpQuarkProduct", "recojet_isD[0] * recojet_isU[1]")
        df = df.Define("JetFlavor_DownUpQuarkProduct_Logit", f"TMath::Log((JetFlavor_DownUpQuarkProduct + {epsilon}) / (1.0 - JetFlavor_DownUpQuarkProduct + {epsilon}))")
        
        df = df.Define("JetFlavor_StrangeCharmQuarkProduct", "recojet_isS[0] * recojet_isC[1]")
        df = df.Define("JetFlavor_StrangeCharmQuarkProduct_Logit", f"TMath::Log((JetFlavor_StrangeCharmQuarkProduct + {epsilon}) / (1.0 - JetFlavor_StrangeCharmQuarkProduct + {epsilon}))")
        
        df = df.Define("JetFlavor_CharmStrangeQuarkProduct", "recojet_isC[0] * recojet_isS[1]")
        df = df.Define("JetFlavor_CharmStrangeQuarkProduct_Logit", f"TMath::Log((JetFlavor_CharmStrangeQuarkProduct + {epsilon}) / (1.0 - JetFlavor_CharmStrangeQuarkProduct + {epsilon}))")

        return df

    # Mandatory: output function
    @staticmethod
    def output():
        '''
        Output variables which will be saved to output root file.
        '''
        branch_list = [
            # Photon variables
            "IsolatedPhoton_P",
            "IsolatedPhoton_Phi",
            "IsolatedPhoton_Theta",
            "IsolatedPhoton_E",
            "IsolatedPhoton_CosTheta",
            "IsolatedPhoton_CosPhi",
            "N_IsolatedPhotons",

            # Electron variables
            "IsolatedElectron_P",
            "IsolatedElectron_Phi",
            "IsolatedElectron_Theta",
            "IsolatedElectron_E",
            "IsolatedElectron_CosTheta",
            "IsolatedElectron_CosPhi",
            "N_IsolatedElectrons",
            "IsolatedElectron_M",

            # Missing quantities variables (Neutrino)
            "MissingQuantities_P",
            "MissingQuantities_E",
            "MissingQuantities_Pt",
            "MissingQuantities_Theta",
            "MissingQuantities_Phi",
            "MissingQuantities_CosTheta",
            "MissingQuantities_CosPhi",
            "MissingQuantities_M",

            # Jet clustering variables
            "JetClustering_d23",
            "JetClustering_d34",

            # Jet constituent counts
            "Jet1_NConstituents",
            "Jet2_NConstituents",
            
            # Jet 1 kinematics
            "Jet1_P",
            "Jet1_Phi",
            "Jet1_M",
            "Jet1_E",
            "Jet1_Theta",
            "Jet1_CosTheta",
            "Jet1_CosPhi",

            # Jet 2 kinematics
            "Jet2_P",
            "Jet2_Phi",
            "Jet2_M",
            "Jet2_E",
            "Jet2_Theta",
            "Jet2_CosTheta",
            "Jet2_CosPhi",

            # Jet energy and charge
            "Jets_MaxEnergy",
            "Jets_MinEnergy",
            "Jet1_Charge",
            "Jet2_Charge",

            # Leptonic W Boson (Electron + Neutrino) kinematics
            "LeptonicW_P",
            "LeptonicW_Phi",
            "LeptonicW_CosPhi",
            "LeptonicW_Theta",
            "LeptonicW_CosTheta",
            "LeptonicW_InvariantMass",
            
            # Hadronic W Boson (Jet1 + Jet2) kinematics
            "HadronicW_P",
            "HadronicW_Phi",
            "HadronicW_CosPhi",
            "HadronicW_Theta",
            "HadronicW_CosTheta",
            "HadronicW_InvariantMass",
            
            # Angular separations between the two W bosons
            "WW_DeltaR",
            "WW_DeltaPhi",
            "WW_DeltaTheta",
            "WW_DeltaInvariantMass",
            "WW_Angle",
            "WW_CosAngle",

            # Event-level masses
            "Event_InvariantMass",
            #"DiJet_InvariantMass",

            # On-shell and off-shell mass variables
            "System_OnShellMass",
            "System_OffShellMass",

            "System_MaxCosTheta",
            "System_MinCosTheta",

            "System_MaxCosPhi",
            "System_MinCosPhi",

            #"System_MaxCosP",
            #"System_MinCosP",

            # MELA angular variables
            "MELA_Phi",
            "MELA_CosPhi",
            "MELA_Phi1",
            "MELA_CosPhi1",
            "MELA_PhiStar",
            "MELA_CosPhiStar",
            "MELA_ThetaStar",
            "MELA_CosThetaStar",
            "MELA_Theta1",
            "MELA_CosTheta1",
            "MELA_Theta2",
            "MELA_CosTheta2",

            # Event shape variables
            "EventShape_Planarity",
            "EventShape_Aplanarity",
            "EventShape_Sphericity",
            "EventShape_Asphericity",

            # Jet flavor tagging scores - Products
            # "JetFlavor_GluonProduct",
            # "JetFlavor_UpQuarkProduct",
            # "JetFlavor_DownQuarkProduct",
            # "JetFlavor_StrangeQuarkProduct",
            # "JetFlavor_CharmQuarkProduct",
            # "JetFlavor_BottomQuarkProduct",
            # "JetFlavor_TauProduct",
            # "JetFlavor_UpDownQuarkProduct",
            # "JetFlavor_DownUpQuarkProduct",
            # "JetFlavor_StrangeCharmQuarkProduct",
            # "JetFlavor_CharmStrangeQuarkProduct",
            
            # Jet flavor tagging scores - Logits
            # "JetFlavor_GluonProduct_Logit",
            "JetFlavor_UpQuarkProduct_Logit",
            "JetFlavor_DownQuarkProduct_Logit",
            "JetFlavor_StrangeQuarkProduct_Logit",
            "JetFlavor_CharmQuarkProduct_Logit",
            "JetFlavor_BottomQuarkProduct_Logit",
            "JetFlavor_TauProduct_Logit",
            "JetFlavor_UpDownQuarkProduct_Logit",
            "JetFlavor_DownUpQuarkProduct_Logit",
            "JetFlavor_StrangeCharmQuarkProduct_Logit",
            "JetFlavor_CharmStrangeQuarkProduct_Logit",
        ]

        return branch_list
