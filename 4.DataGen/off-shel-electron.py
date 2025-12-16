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
            "wzp6_ee_Henueqq_ecm125": {"fraction": 10},
            "wzp6_ee_Hqqenue_ecm125": {"fraction": 10},
            "wzp6_ee_Hmunumuqq_ecm125": {"fraction": 10},
            "wzp6_ee_Hqqmunumu_ecm125": {"fraction": 10},
            "wzp6_ee_Htaunutauqq_ecm125": {"fraction": 10},
            "wzp6_ee_Hqqtaunutau_ecm125": {"fraction": 10},
            "wzp6_ee_taunutauqq_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_tautauqq_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_enueqq_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_eeqq_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_munumuqq_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_mumuqq_ecm125": {"fraction": 1,'chunks': 10},
            # Fully leptonic Processes
            #"wzp6_ee_Htautau_ecm125": {"fraction": 1},
            #"wzp6_ee_Hllnunu_ecm125": {"fraction": 1},
            "wzp6_ee_eenunu_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_mumununu_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_tautaununu_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_l1l2nunu_ecm125": {"fraction": 1,'chunks': 10},
            "wzp6_ee_tautau_ecm125": {"fraction": 1,'chunks': 10},
            # Fully hadronic Processes
            #"wzp6_ee_Hgg_ecm125": {"fraction": 1},
            #"wzp6_ee_Hbb_ecm125": {"fraction": 1},
            "wzp6_ee_qq_ecm125": {"fraction": 1,'chunks': 10},
            "p8_ee_ZZ_4tau_ecm125": {"fraction": 1,'chunks': 10},
}

outputDir = "/eos/experiment/fcc/ee/analyses/case-studies/higgs/electron_yukawa/DataGenReduced/off-shell-electron/"
inputDir = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA"
nCPUS = -1
includePaths = ["../src/functions.h", "../src/GEOFunctions.h", "../src/MELAFunctions.h","../src/SortJets.h" ]

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
    """RDFanalysis class: defines the transformations applied to the input dataframe."""

    @staticmethod
    def analysers(df):
        """Define aliases, variables, clustering, tagger inference, and filters."""

        # Aliases
        df = df.Alias("Particle0", "Particle#0.index")
        df = df.Alias("Particle1", "Particle#1.index")
        df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
        df = df.Alias("Electron0", "Electron#0.index")
        df = df.Alias("Muon0", "Muon#0.index")
        df = df.Alias("Photon0", "Photon#0.index")
        df = df.Alias("Jet2", "Jet#2.index")


        # Missing energy cut 
        df = df.Define("MissingE_4p", "FCCAnalyses::ReconstructedParticle::get_tlv(MissingET)")
        df = df.Define("Missing_Pt", "MissingE_4p[0].Pt()")
        df = df.Filter("Missing_Pt > 3  ")

        # Photons and charged hadrons
        df = df.Define("photons_all", "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)")
        df = df.Define("photons", "FCCAnalyses::ReconstructedParticle::sel_p(20)(photons_all)")
        df = df.Define(
            "ChargedHadrons",
            "ReconstructedParticle2MC::selRP_ChargedHadrons( MCRecoAssociations0,MCRecoAssociations1,ReconstructedParticles,Particle)",
        )

        # Leptons
        df = df.Define("electrons_all", "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)")
        df = df.Define("muons_all", "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)")
        df = df.Define("electrons", "FCCAnalyses::ReconstructedParticle::sel_p(0)(electrons_all)")
        df = df.Define("muons", "FCCAnalyses::ReconstructedParticle::sel_p(0)(muons_all)")

        # Isolation using JSON-configurable parameters
        df = df.Define(
            "electrons_iso",
            f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(electrons, ChargedHadrons)",
        )
        df = df.Define(
            "electrons_sel_iso",
            f"FCCAnalyses::ZHfunctions::sel_iso({selection})(electrons, electrons_iso)",
        )

        df = df.Define(
            "muons_iso",
            f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(muons, ChargedHadrons)",
        )
        df = df.Define(
            "muons_sel_iso",
            f"FCCAnalyses::ZHfunctions::sel_iso({selection})(muons, muons_iso)",
        )

        df = df.Define("photons_iso", f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(photons, ChargedHadrons)")
        df = df.Define("photons_sel_iso", f"FCCAnalyses::ZHfunctions::sel_iso({selection})(photons, photons_iso)")

        # Electron variables:
        df = df.Define("IsoMuonNum", "muons_sel_iso.size()")
        df = df.Define("Iso_Electrons_No", "electrons_sel_iso.size()")

        # Require exactly one isolated electron
        df = df.Filter(" Iso_Electrons_No == 1 ")
        df = df.Filter(" IsoMuonNum == 0 ")

        # Isolated lepton 4-vectors
        df = df.Define("IsoElectron_4p", "FCCAnalyses::ReconstructedParticle::get_tlv(electrons_sel_iso)")
        df = df.Define("IsoMuon_4p", "FCCAnalyses::ReconstructedParticle::get_tlv(muons_sel_iso)")

        

        # Isolated photons info
        df = df.Define("IsoPhotons_4p", "FCCAnalyses::ReconstructedParticle::get_tlv(photons_sel_iso)")
        df = df.Define("Iso_Photon_Phi", "IsoPhotons_4p[0].Phi()")
        df = df.Define("Iso_Photon_Theta", "IsoPhotons_4p[0].Theta()")
        df = df.Define("Iso_Photon_E", "IsoPhotons_4p[0].E()")
        df = df.Define("Iso_Photon_CosTheta", "IsoPhotons_4p[0].CosTheta()")
        df = df.Define("Iso_Photon_CosPhi", "cos(Iso_Photon_Phi)")
        df = df.Define("Iso_Photons_No", "photons_sel_iso.size()")

        # Isolated electron components & properties
        df = df.Define("IsoElectron_3p", "IsoElectron_4p[0].Vect()")
        df = df.Define("Iso_Electron_P", "IsoElectron_4p[0].P()")
        df = df.Define("Iso_Electron_Phi", "IsoElectron_4p[0].Phi()")
        df = df.Define("Iso_Electron_Theta", "IsoElectron_4p[0].Theta()")
        df = df.Define("Iso_Electron_E", "IsoElectron_4p[0].E()")
        df = df.Define("Iso_Electron_CosTheta", "IsoElectron_4p[0].CosTheta()")
        df = df.Define("Iso_Electron_CosPhi", "TMath::Cos(Iso_Electron_Phi)")
        df = df.Define("Iso_Electron_Charge", "FCCAnalyses::ReconstructedParticle::get_charge(electrons_sel_iso)[0]")

        # Missing energy repeated derivatives (kept for compatibility)
        df = df.Define("Missing_P", "MissingE_4p[0].P()")
        df = df.Define("Missing_E", "MissingE_4p[0].E()")

        # Combined masses
        df = df.Define("EnuM", " (MissingE_4p[0]+IsoElectron_4p[0]).M()")

        # Create collections with particles removed
        df = df.Define(
            "ReconstructedParticles_nophotons",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, photons)",
        )
        df = df.Define(
            "ReconstructedParticlesNoElectrons",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles_nophotons,electrons)",
        )
        df = df.Define(
            "ReconstructedParticlesNoleptonsNoPhotons",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticlesNoElectrons,muons)",
        )

        # Jet clustering setup
        global jetClusteringHelper
        global jetFlavourHelper

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

        jetClusteringHelper = ExclusiveJetClusteringHelper(collections_noleptons_nophotons["PFParticles"], 2)
        df = jetClusteringHelper.define(df)

        # Jet flavour tagger
        jetFlavourHelper = JetFlavourHelper(
            collections_noleptons_nophotons,
            jetClusteringHelper.jets,
            jetClusteringHelper.constituents,
        )
        df = jetFlavourHelper.define(df)

        

        # Filters and jet definitions
        df = df.Filter("event_njet > 1")
        df = df.Define("Jets_p4", "JetConstituentsUtils::compute_tlv_jets({})".format(jetClusteringHelper.jets))
        df = df.Define("Jets_InMa", "JetConstituentsUtils::InvariantMass(Jets_p4[0], Jets_p4[1])")
        df = df.Filter("Jets_InMa > 52.85")
        

        # Tagger inference
        df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)

        jetClusteringHelper_N2 = ExclusiveJetClusteringHelper("ReconstructedParticlesNoleptonsNoPhotons", 2, "N2")
        df = jetClusteringHelper_N2.define(df)

        df = df.Define("d23", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 2))")
        df = df.Define("d34", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 3))")
        df = df.Define("Jets_charge", "JetConstituentsUtils::get_charge({})".format(jetClusteringHelper.constituents))
        df = df.Define("Jet_nconst0", "jet_nconst[0]")
        df = df.Define("Jet_nconst1", "jet_nconst[1]")
        df = df.Filter("Jet_nconst0 > 2")
        df = df.Filter("Jet_nconst1 > 2")


        # Jet kinematics (jet 1)
        df = df.Define("Jet1_P3", "Jets_p4[0].Vect()")
        df = df.Define("Jet1_P", "Jets_p4[0].P()")
        df = df.Define("Jet1_Eta", "Jets_p4[0].Eta()")
        df = df.Define("Jet1_Phi", "Jets_p4[0].Phi()")
        df = df.Define("Jet1_M", "Jets_p4[0].M()")
        df = df.Define("Jet1_E", "Jets_p4[0].E()")
        df = df.Define("Jet1_Theta", "Jets_p4[0].Theta()")
        df = df.Define("Jet1_CosTheta", "Jets_p4[0].CosTheta()")
        df = df.Define("Jet1_CosPhi", "TMath::Cos(Jet1_Phi)")

        # Jet kinematics (jet 2)
        df = df.Define("Jet2_P3", "Jets_p4[1].Vect()")
        df = df.Define("Jet2_P", "Jets_p4[1].P()")
        df = df.Define("Jet2_Eta", "Jets_p4[1].Eta()")
        df = df.Define("Jet2_Phi", "Jets_p4[1].Phi()")
        df = df.Define("Jet2_M", "Jets_p4[1].M()")
        df = df.Define("Jet2_E", "Jets_p4[1].E()")
        df = df.Define("Jet2_Theta", "Jets_p4[1].Theta()")
        df = df.Define("Jet2_CosTheta", "Jets_p4[1].CosTheta()")
        df = df.Define("Jet2_CosPhi", "TMath::Cos(Jet2_Phi)")

        # Jet comparisons and event-level values
        df = df.Define("Max_JetsP", "TMath::Max(Jets_p4[0].Pt(),Jets_p4[1].P())")
        df = df.Define("Min_JetsP", "TMath::Min(Jets_p4[0].Pt(),Jets_p4[1].P())")
        df = df.Define("Max_JetsE", "TMath::Max(Jets_p4[0].E(),Jets_p4[1].E())")
        df = df.Define("Min_JetsE", "TMath::Min(Jets_p4[0].E(),Jets_p4[1].E())")
        df = df.Define("Jet1_charge", "ROOT::VecOps::Sum(Jets_charge[0])")
        df = df.Define("Jet2_charge", "ROOT::VecOps::Sum(Jets_charge[1])")

        # Jet angular/kinematic relations
        df = df.Define("Jets_delR", "Jets_p4[0].DeltaR(Jets_p4[1])")
        df = df.Define("ILjet1_delR", "IsoElectron_4p[0].DeltaR(Jets_p4[0])")
        df = df.Define("ILjet2_delR", "IsoElectron_4p[0].DeltaR(Jets_p4[1])")
        df = df.Define("Max_DelRLJets", "TMath::Max(ILjet1_delR,ILjet2_delR)")
        df = df.Define("Min_DelRLJets", "TMath::Min(ILjet1_delR,ILjet2_delR)")

        df = df.Define("Jets_delphi", "Jets_p4[0].DeltaPhi(Jets_p4[1])")
        df = df.Define("ILjet1_delphi", "IsoElectron_4p[0].DeltaPhi(Jets_p4[0])")
        df = df.Define("ILjet2_delphi", "IsoElectron_4p[0].DeltaPhi(Jets_p4[1])")
        df = df.Define("Max_DelPhiLJets", "TMath::Max(ILjet1_delphi,ILjet2_delphi)")
        df = df.Define("Min_DelPhiLJets", "TMath::Min(ILjet1_delphi,ILjet2_delphi)")

        df = df.Define("Jets_deltheta", "Jet1_Theta - Jet2_Theta")
        df = df.Define("Jets_angle", "Jets_p4[0].Angle(Jets_p4[1].Vect())")
        df = df.Define("ILjet1_angle", "IsoElectron_4p[0].Angle(Jets_p4[0].Vect())")
        df = df.Define("ILjet2_angle", "IsoElectron_4p[0].Angle(Jets_p4[1].Vect())")
        df = df.Define("Jets_cosangle", "TMath::Cos(Jets_angle)")
        df = df.Define("ILjet1_cosangle", "TMath::Cos(ILjet1_angle)")
        df = df.Define("ILjet2_cosangle", "TMath::Cos(ILjet2_angle)")
        df = df.Define("Max_CosLJets", "TMath::Max(ILjet1_cosangle,ILjet2_cosangle)")
        df = df.Define("Min_CosLJets", "TMath::Min(ILjet1_cosangle,ILjet2_cosangle)")

        df = df.Define("Event_IM", "(Jets_p4[0] + Jets_p4[1] + MissingE_4p[0] + IsoElectron_4p[0]).M()")

        # Masses and combinations
        df = df.Define("LJJ_M", "(Jets_p4[0] + Jets_p4[1] + IsoElectron_4p[0]).M()")
        df = df.Define("LJ1_M", "(Jets_p4[0] + IsoElectron_4p[0]).M()")
        df = df.Define("LJ2_M", "(Jets_p4[1] + IsoElectron_4p[0]).M()")
        df = df.Define("JJ_M", "(Jets_p4[0] + Jets_p4[1]).M()")
        df = df.Define("JJ_E", "(Jets_p4[0] + Jets_p4[1]).E()")


        df = df.Define("ljj_Phi", "(Jets_p4[0] + Jets_p4[1] + IsoElectron_4p[0]).Phi()")
        df = df.Define("jj_Phi", "(Jets_p4[0] + Jets_p4[1]).Phi()")

        df = df.Define("Wl_M", "(IsoElectron_4p[0] + MissingE_4p[0]).M()")
        df = df.Define("Wl_Theta", "(IsoElectron_4p[0] + MissingE_4p[0]).Theta()")

        df = df.Define("Shell_M", "TMath::Max((Jets_p4[0]+Jets_p4[1]).M(),Wl_M)")
        df = df.Define("Off_M", "TMath::Min((Jets_p4[0]+Jets_p4[1]).M(),Wl_M)")
        df = df.Define("CosTheta_MaxjjW", "TMath::Cos(TMath::Max((Jets_p4[0]+Jets_p4[1]).Theta(),Wl_Theta))")
        df = df.Define("CosTheta_MinjjW", "TMath::Cos(TMath::Min((Jets_p4[0]+Jets_p4[1]).Theta(),Wl_Theta))")
        df = df.Define("expD", "125.-MissingE_4p[0].E()-IsoElectron_4p[0].E()-Jets_p4[0].E()-Jets_p4[1].E()")

        # MELA Variables
        df = df.Define(
            "mela",
            "FCCAnalyses::MELA::MELACalculator::mela(Jets_p4[0],Jets_p4[1],MissingE_4p[0],IsoElectron_4p[0], Iso_Electron_Charge,Jet1_charge,Jet2_charge)",
        )
        df = df.Define("Phi", " mela.phi")
        df = df.Define("CosPhi", " mela.cosPhi")
        df = df.Define("Phi1", " mela.phi1")
        df = df.Define("CosPhi1", " mela.cosPhi1")
        df = df.Define("PhiStar", " mela.phiStar")
        df = df.Define("CosPhiStar", " mela.cosPhiStar")
        df = df.Define("ThetaStar", " mela.thetaStar")
        df = df.Define("CosThetaStar", "mela.cosThetaStar")
        df = df.Define("Theta1", " mela.theta1")
        df = df.Define("Costheta1", " mela.cosTheta1")
        df = df.Define("Theta2", " mela.theta2")
        df = df.Define("Costheta2", " mela.cosTheta2")

        # Event-shape variables
        df = df.Define("Planarity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculatePlanarity(Jet1_P3,Jet2_P3,IsoElectron_3p)")
        df = df.Define("APlanarity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAplanarity(Jet1_P3,Jet2_P3,IsoElectron_3p)")
        df = df.Define("Sphericity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateSphericity(Jet1_P3,Jet2_P3,IsoElectron_3p)")
        df = df.Define("ASphericity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAsphericity(Jet1_P3,Jet2_P3,IsoElectron_3p)")


        # Sums of scores
        df = df.Define("scoreSumUD", "recojet_isU[0]+recojet_isD[1]")
        df = df.Define("scoreSumDU", "recojet_isD[0]+recojet_isU[1]")

        df = df.Define("scoreSumSC", "recojet_isS[0]+recojet_isC[1]")
        df = df.Define("scoreSumCS", "recojet_isC[0]+recojet_isS[1]")

        df = df.Define("scoreProdUD", "recojet_isU[0]*recojet_isD[1]")
        df = df.Define("scoreProdDU", "recojet_isD[0]*recojet_isU[1]")

        df = df.Define("scoreProdSC", "recojet_isS[0]*recojet_isC[1]")
        df = df.Define("scoreProdCS", "recojet_isC[0]*recojet_isS[1]")



        df = df.Define("scoreSumB", "recojet_isB[0]+recojet_isB[1]")
        df = df.Define("scoreSumT", "recojet_isTAU[0]+recojet_isTAU[1]")

        # Products of scores
        df = df.Define("scoreMultiplyG", "recojet_isG[0]*recojet_isG[1]")
        df = df.Define("scoreMultiplyU", "recojet_isU[0]*recojet_isU[1]")
        df = df.Define("scoreMultiplyS", "recojet_isS[0]*recojet_isS[1]")
        df = df.Define("scoreMultiplyC", "recojet_isC[0]*recojet_isC[1]")
        df = df.Define("scoreMultiplyB", "recojet_isB[0]*recojet_isB[1]")
        df = df.Define("scoreMultiplyT", "recojet_isTAU[0]*recojet_isTAU[1]")
        df = df.Define("scoreMultiplyD", "recojet_isD[0]*recojet_isD[1]")

        return df

    @staticmethod
    def output():
        """Return a list of all branches defined in analysers()."""
        branchList = [

            "Iso_Photon_Phi",
            "Iso_Photon_Theta",
            "Iso_Photon_E",
            "Iso_Photon_CosTheta",
            "Iso_Photon_CosPhi",
            "Iso_Photons_No",

            "Iso_Electron_P",
            "Iso_Electron_Phi",
            "Iso_Electron_Theta",
            "Iso_Electron_E",
            "Iso_Electron_CosTheta",
            "Iso_Electron_CosPhi",
            "Iso_Electrons_No",

            "Missing_P",
            "Missing_E",
            "Missing_Pt",

            "EnuM",
            "Jets_InMa",

            "d23",
            "d34",

            "Jet_nconst0",
            "Jet_nconst1",
            
            "Jet1_P",
            "Jet1_Phi",
            "Jet1_M",
            "Jet1_E",
            "Jet1_Theta",
            "Jet1_CosTheta",
            "Jet1_CosPhi",

            "Jet2_P",
            "Jet2_Phi",
            "Jet2_M",
            "Jet2_E",
            "Jet2_Theta",
            "Jet2_CosTheta",
            "Jet2_CosPhi",

            "Max_JetsE",
            "Min_JetsE",
            "Jet1_charge",
            "Jet2_charge",

            "Jets_delR",
            "ILjet1_delR",
            "ILjet2_delR",
            "Max_DelRLJets",
            "Min_DelRLJets",

            "Jets_delphi",
            "ILjet1_delphi",
            "ILjet2_delphi",
            "Max_DelPhiLJets",
            "Min_DelPhiLJets",

            "Jets_deltheta",
            "Jets_angle",
            "ILjet1_angle",
            "ILjet2_angle",
            "Jets_cosangle",
            "ILjet1_cosangle",
            "ILjet2_cosangle",
            "Max_CosLJets",
            "Min_CosLJets",

            "Event_IM",
            "LJJ_M",
            "LJ1_M",
            "LJ2_M",
            "JJ_M",
            "JJ_E",

            "ljj_Phi",
            "jj_Phi",

            "Wl_M",
            "Wl_Theta",
            "Shell_M",
            "Off_M",
            "CosTheta_MaxjjW",
            "CosTheta_MinjjW",
            "expD",

            "Phi",
            "CosPhi",
            "Phi1",
            "CosPhi1",
            "PhiStar",
            "CosPhiStar",
            "ThetaStar",
            "CosThetaStar",
            "Theta1",
            "Costheta1",
            "Theta2",
            "Costheta2",
            "Planarity",
            "APlanarity",
            "Sphericity",
            "ASphericity",

            "scoreMultiplyG",
            "scoreMultiplyU",
            "scoreMultiplyS",
            "scoreMultiplyC",
            "scoreMultiplyB",
            "scoreMultiplyT",
            "scoreMultiplyD",

            "scoreSumUD",
            "scoreSumDU",
            "scoreSumSC",
            "scoreSumCS",
            "scoreSumB",
            "scoreSumT",

            "scoreProdUD",
            "scoreProdDU",
            "scoreProdSC",
            "scoreProdCS",







        ]
        return branchList
    
