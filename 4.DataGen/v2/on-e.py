'''
Analysis for electron Yukawa coupling measurement in semileptonic channel.
This analysis stage runs on HTCondor.
'''
from argparse import ArgumentParser
import copy
import urllib.request
import os


# Mandatory: Analysis class where the user defines the operations on the dataframe
class Analysis():
    '''
    Electron Yukawa coupling analysis in semileptonic WW channel.
    '''
    def __init__(self, cmdline_args):
        # Parse additional arguments
        parser = ArgumentParser(
            description='Additional analysis arguments',
            usage='Provided after "--"')
        parser.add_argument('--dr-min', default=0.01, type=float,
                            help='Minimum delta R for cone isolation.')
        parser.add_argument('--dr-max', default=0.2, type=float,
                            help='Maximum delta R for cone isolation.')
        parser.add_argument('--isolation-cut', default=0.2, type=float,
                            help='Isolation selection threshold.')
        
        self.ana_args, _ = parser.parse_known_args(cmdline_args['unknown'])

        # Mandatory: List of processes used in the analysis
        self.process_list = {
            # Semileptonic processes
            "wzp6_ee_Henueqq_ecm125": {"fraction": 1},
            "wzp6_ee_Hqqenue_ecm125": {"fraction": 1},
            "wzp6_ee_Hmunumuqq_ecm125": {"fraction": 1},
            "wzp6_ee_Hqqmunumu_ecm125": {"fraction": 1},
            "wzp6_ee_Htaunutauqq_ecm125": {"fraction": 1},
            "wzp6_ee_Hqqtaunutau_ecm125": {"fraction": 1},
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

        # Mandatory: Production tag
        self.prod_tag = 'FCCee/winter2023/IDEA/'

        # Optional: output directory
        self.output_dir = '/eos/user/h/hfatehi/electron_yukawa/on/stage1'

        # Optional: analysis name
        self.analysis_name = 'Electron Yukawa Analysis'

        # Optional: number of threads
        self.n_threads = 32
        self.run_batch = True
        self.eos_type = 'eospublic'

        # Optional: batch queue name
        self.batch_queue = 'testmatch'

        # Optional: computing account
        self.comp_group = 'group_u_FCC.local_gen'

        # Optional: output directory on eos
        self.output_dir_eos = '/eos/experiment/fcc/ee/analyses/case-studies/higgs/electron_yukawa/FinalRound/on-shell-electron/'

        # Optional: include paths for custom functions
        self.include_paths = ["../../src/functions.h", "../../src/GEOFunctions.h", 
                             "../../src/MELAFunctions.h", "../../src/SortJets.h"]

        # Jet flavor tagging model setup
        self.model_name = "fccee_flavtagging_edm4hep_wc"
        self.url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
        self.model_dir = "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_7classes_12_04_2023/"
        
        url_preproc = "{}/{}.json".format(self.url_model_dir, self.model_name)
        url_model = "{}/{}.onnx".format(self.url_model_dir, self.model_name)
        local_preproc = "{}/{}.json".format(self.model_dir, self.model_name)
        local_model = "{}/{}.onnx".format(self.model_dir, self.model_name)
        
        self.weaver_preproc = self.get_file_path(url_preproc, local_preproc)
        self.weaver_model = self.get_file_path(url_model, local_model)

        # Helper objects (will be initialized in analyzers)
        self.jetFlavourHelper = None
        self.jetClusteringHelper = None

    @staticmethod
    def get_file_path(url, filename):
        """Return local file path if exists else download from url and return basename."""
        if os.path.exists(filename):
            return os.path.abspath(filename)
        else:
            urllib.request.urlretrieve(url, os.path.basename(url))
            return os.path.basename(url)

    # Mandatory: analyzers function to define the analysis graph
    def analyzers(self, dframe):
        '''
        Analysis graph for electron Yukawa coupling measurement.
        '''
        from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
        from addons.FastJet.jetClusteringHelper import ExclusiveJetClusteringHelper

        # Get isolation parameters
        drmin = self.ana_args.dr_min
        drmax = self.ana_args.dr_max
        selection = self.ana_args.isolation_cut

        dframe2 = dframe

        # ===========================
        # Aliases
        # ===========================
        dframe2 = dframe2.Alias("Particle0", "Particle#0.index")
        dframe2 = dframe2.Alias("Particle1", "Particle#1.index")
        dframe2 = dframe2.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        dframe2 = dframe2.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
        dframe2 = dframe2.Alias("Electron0", "Electron#0.index")
        dframe2 = dframe2.Alias("Muon0", "Muon#0.index")
        dframe2 = dframe2.Alias("Photon0", "Photon#0.index")
        dframe2 = dframe2.Alias("Jet2", "Jet#2.index")

        # ===========================
        # Missing energy variables
        # ===========================
        dframe2 = dframe2.Define("MissingQuantities_4Vec", "FCCAnalyses::ReconstructedParticle::get_tlv(MissingET)")
        dframe2 = dframe2.Define("MissingQuantities_Pt", "MissingQuantities_4Vec[0].Pt()")
        dframe2 = dframe2.Filter("MissingQuantities_Pt > 3")
        dframe2 = dframe2.Define("MissingQuantities_P", "MissingQuantities_4Vec[0].P()")
        dframe2 = dframe2.Define("MissingQuantities_E", "MissingQuantities_4Vec[0].E()")
        dframe2 = dframe2.Define("MissingQuantities_Theta", "MissingQuantities_4Vec[0].Theta()")
        dframe2 = dframe2.Define("MissingQuantities_Phi", "MissingQuantities_4Vec[0].Phi()")
        dframe2 = dframe2.Define("MissingQuantities_CosTheta", "MissingQuantities_4Vec[0].CosTheta()")
        dframe2 = dframe2.Define("MissingQuantities_CosPhi", "TMath::Cos(MissingQuantities_Phi)")

        # ===========================
        # Photons and charged hadrons
        # ===========================
        dframe2 = dframe2.Define("Photons_All", "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)")
        dframe2 = dframe2.Define("Photons_Selected", "FCCAnalyses::ReconstructedParticle::sel_p(20)(Photons_All)")
        dframe2 = dframe2.Define(
            "ChargedHadrons",
            "ReconstructedParticle2MC::selRP_ChargedHadrons(MCRecoAssociations0,MCRecoAssociations1,ReconstructedParticles,Particle)",
        )

        # ===========================
        # Leptons
        # ===========================
        dframe2 = dframe2.Define("Electrons_All", "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)")
        dframe2 = dframe2.Define("Muons_All", "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)")
        dframe2 = dframe2.Define("Electrons_PreSelection", "FCCAnalyses::ReconstructedParticle::sel_p(0)(Electrons_All)")
        dframe2 = dframe2.Define("Muons_PreSelection", "FCCAnalyses::ReconstructedParticle::sel_p(0)(Muons_All)")

        # ===========================
        # Isolation
        # ===========================
        dframe2 = dframe2.Define(
            "Electrons_IsolationValue",
            f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(Electrons_PreSelection, ChargedHadrons)",
        )
        dframe2 = dframe2.Define(
            "Electrons_Isolated",
            f"FCCAnalyses::ZHfunctions::sel_iso({selection})(Electrons_PreSelection, Electrons_IsolationValue)",
        )

        dframe2 = dframe2.Define(
            "Muons_IsolationValue",
            f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(Muons_PreSelection, ChargedHadrons)",
        )
        dframe2 = dframe2.Define(
            "Muons_Isolated",
            f"FCCAnalyses::ZHfunctions::sel_iso({selection})(Muons_PreSelection, Muons_IsolationValue)",
        )

        dframe2 = dframe2.Define(
            "Photons_IsolationValue",
            f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(Photons_Selected, ChargedHadrons)"
        )
        dframe2 = dframe2.Define(
            "Photons_Isolated",
            f"FCCAnalyses::ZHfunctions::sel_iso({selection})(Photons_Selected, Photons_IsolationValue)"
        )

        # ===========================
        # Lepton counting and selection
        # ===========================
        dframe2 = dframe2.Define("N_IsolatedMuons", "Muons_Isolated.size()")
        dframe2 = dframe2.Define("N_IsolatedElectrons", "Electrons_Isolated.size()")

        # Require exactly one isolated electron and no isolated muons
        dframe2 = dframe2.Filter("N_IsolatedElectrons == 1")
        dframe2 = dframe2.Filter("N_IsolatedMuons == 0")

        # ===========================
        # Isolated particle 4-vectors
        # ===========================
        dframe2 = dframe2.Define("IsolatedElectron_4Vec", "FCCAnalyses::ReconstructedParticle::get_tlv(Electrons_Isolated)")
        dframe2 = dframe2.Define("IsolatedMuon_4Vec", "FCCAnalyses::ReconstructedParticle::get_tlv(Muons_Isolated)")
        dframe2 = dframe2.Define("IsolatedPhotons_4Vec", "FCCAnalyses::ReconstructedParticle::get_tlv(Photons_Isolated)")

        # ===========================
        # Isolated photon properties
        # ===========================
        dframe2 = dframe2.Define("IsolatedPhoton_Phi", "IsolatedPhotons_4Vec[0].Phi()")
        dframe2 = dframe2.Define("IsolatedPhoton_Theta", "IsolatedPhotons_4Vec[0].Theta()")
        dframe2 = dframe2.Define("IsolatedPhoton_E", "IsolatedPhotons_4Vec[0].E()")
        dframe2 = dframe2.Define("IsolatedPhoton_CosTheta", "IsolatedPhotons_4Vec[0].CosTheta()")
        dframe2 = dframe2.Define("IsolatedPhoton_CosPhi", "TMath::Cos(IsolatedPhoton_Phi)")
        dframe2 = dframe2.Define("N_IsolatedPhotons", "Photons_Isolated.size()")

        # ===========================
        # Isolated electron properties
        # ===========================
        dframe2 = dframe2.Define("IsolatedElectron_3Vec", "IsolatedElectron_4Vec[0].Vect()")
        dframe2 = dframe2.Define("IsolatedElectron_P", "IsolatedElectron_4Vec[0].P()")
        dframe2 = dframe2.Define("IsolatedElectron_Phi", "IsolatedElectron_4Vec[0].Phi()")
        dframe2 = dframe2.Define("IsolatedElectron_Theta", "IsolatedElectron_4Vec[0].Theta()")
        dframe2 = dframe2.Define("IsolatedElectron_E", "IsolatedElectron_4Vec[0].E()")
        dframe2 = dframe2.Define("IsolatedElectron_CosTheta", "IsolatedElectron_4Vec[0].CosTheta()")
        dframe2 = dframe2.Define("IsolatedElectron_CosPhi", "TMath::Cos(IsolatedElectron_Phi)")
        dframe2 = dframe2.Define("IsolatedElectron_Charge", "FCCAnalyses::ReconstructedParticle::get_charge(Electrons_Isolated)[0]")

        # ===========================
        # Create collections with particles removed
        # ===========================
        dframe2 = dframe2.Define(
            "RecoParticles_NoPhotons",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, Photons_Selected)",
        )
        dframe2 = dframe2.Define(
            "RecoParticles_NoPhotonsNoElectrons",
            "FCCAnalyses::ReconstructedParticle::remove(RecoParticles_NoPhotons, Electrons_PreSelection)",
        )
        dframe2 = dframe2.Define(
            "RecoParticles_NoLeptonsNoPhotons",
            "FCCAnalyses::ReconstructedParticle::remove(RecoParticles_NoPhotonsNoElectrons, Muons_PreSelection)",
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

        collections_noleptons_nophotons = copy.deepcopy(collections)
        collections_noleptons_nophotons["PFParticles"] = "RecoParticles_NoLeptonsNoPhotons"

        self.jetClusteringHelper = ExclusiveJetClusteringHelper(
            collections_noleptons_nophotons["PFParticles"], 2
        )
        dframe2 = self.jetClusteringHelper.define(dframe2)

        # ===========================
        # Jet flavour tagger
        # ===========================
        self.jetFlavourHelper = JetFlavourHelper(
            collections_noleptons_nophotons,
            self.jetClusteringHelper.jets,
            self.jetClusteringHelper.constituents,
        )
        dframe2 = self.jetFlavourHelper.define(dframe2)

        # ===========================
        # Jet filtering and definitions
        # ===========================
        dframe2 = dframe2.Filter("event_njet > 1")
        dframe2 = dframe2.Define("Jets_4Vec", 
                                 "JetConstituentsUtils::compute_tlv_jets({})".format(self.jetClusteringHelper.jets))
        dframe2 = dframe2.Define("DiJet_InvariantMass", "JetConstituentsUtils::InvariantMass(Jets_4Vec[0], Jets_4Vec[1])")
        dframe2 = dframe2.Filter("DiJet_InvariantMass < 52.85")
        dframe2 = dframe2.Filter("DiJet_InvariantMass > 4")

        # ===========================
        # Tagger inference
        # ===========================
        dframe2 = self.jetFlavourHelper.inference(self.weaver_preproc, self.weaver_model, dframe2)

        # ===========================
        # Additional jet clustering
        # ===========================
        jetClusteringHelper_N2 = ExclusiveJetClusteringHelper("RecoParticles_NoLeptonsNoPhotons", 2, "N2")
        dframe2 = jetClusteringHelper_N2.define(dframe2)

        dframe2 = dframe2.Define("JetClustering_d23", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 2))")
        dframe2 = dframe2.Define("JetClustering_d34", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 3))")
        dframe2 = dframe2.Define("Jets_Charge", "JetConstituentsUtils::get_charge({})".format(self.jetClusteringHelper.constituents))
        dframe2 = dframe2.Define("Jet1_NConstituents", "jet_nconst[0]")
        dframe2 = dframe2.Define("Jet2_NConstituents", "jet_nconst[1]")
        dframe2 = dframe2.Filter("Jet1_NConstituents > 2")
        dframe2 = dframe2.Filter("Jet2_NConstituents > 2")

        # ===========================
        # Jet 1 kinematics
        # ===========================
        dframe2 = dframe2.Define("Jet1_3Vec", "Jets_4Vec[0].Vect()")
        dframe2 = dframe2.Define("Jet1_P", "Jets_4Vec[0].P()")
        dframe2 = dframe2.Define("Jet1_Eta", "Jets_4Vec[0].Eta()")
        dframe2 = dframe2.Define("Jet1_Phi", "Jets_4Vec[0].Phi()")
        dframe2 = dframe2.Define("Jet1_M", "Jets_4Vec[0].M()")
        dframe2 = dframe2.Define("Jet1_E", "Jets_4Vec[0].E()")
        dframe2 = dframe2.Define("Jet1_Theta", "Jets_4Vec[0].Theta()")
        dframe2 = dframe2.Define("Jet1_CosTheta", "Jets_4Vec[0].CosTheta()")
        dframe2 = dframe2.Define("Jet1_CosPhi", "TMath::Cos(Jet1_Phi)")

        # ===========================
        # Jet 2 kinematics
        # ===========================
        dframe2 = dframe2.Define("Jet2_3Vec", "Jets_4Vec[1].Vect()")
        dframe2 = dframe2.Define("Jet2_P", "Jets_4Vec[1].P()")
        dframe2 = dframe2.Define("Jet2_Eta", "Jets_4Vec[1].Eta()")
        dframe2 = dframe2.Define("Jet2_Phi", "Jets_4Vec[1].Phi()")
        dframe2 = dframe2.Define("Jet2_M", "Jets_4Vec[1].M()")
        dframe2 = dframe2.Define("Jet2_E", "Jets_4Vec[1].E()")
        dframe2 = dframe2.Define("Jet2_Theta", "Jets_4Vec[1].Theta()")
        dframe2 = dframe2.Define("Jet2_CosTheta", "Jets_4Vec[1].CosTheta()")
        dframe2 = dframe2.Define("Jet2_CosPhi", "TMath::Cos(Jet2_Phi)")

        # ===========================
        # Jet energy and charge
        # ===========================
        dframe2 = dframe2.Define("Jets_MaxEnergy", "TMath::Max(Jets_4Vec[0].E(), Jets_4Vec[1].E())")
        dframe2 = dframe2.Define("Jets_MinEnergy", "TMath::Min(Jets_4Vec[0].E(), Jets_4Vec[1].E())")
        dframe2 = dframe2.Define("Jet1_Charge", "ROOT::VecOps::Sum(Jets_Charge[0])")
        dframe2 = dframe2.Define("Jet2_Charge", "ROOT::VecOps::Sum(Jets_Charge[1])")

        # ===========================
        # PAIR SYSTEM DEFINITIONS
        # ===========================
        
        # Leptonic W Boson (Electron + Neutrino)
        dframe2 = dframe2.Define("LeptonicW_4Vec", "IsolatedElectron_4Vec[0] + MissingQuantities_4Vec[0]")
        dframe2 = dframe2.Define("LeptonicW_P", "LeptonicW_4Vec.P()")
        dframe2 = dframe2.Define("LeptonicW_Phi", "LeptonicW_4Vec.Phi()")
        dframe2 = dframe2.Define("LeptonicW_CosPhi", "TMath::Cos(LeptonicW_Phi)")
        dframe2 = dframe2.Define("LeptonicW_Theta", "LeptonicW_4Vec.Theta()")
        dframe2 = dframe2.Define("LeptonicW_CosTheta", "LeptonicW_4Vec.CosTheta()")
        dframe2 = dframe2.Define("LeptonicW_InvariantMass", "LeptonicW_4Vec.M()")
        
        # Hadronic W Boson (Jet1 + Jet2)
        dframe2 = dframe2.Define("HadronicW_4Vec", "Jets_4Vec[0] + Jets_4Vec[1]")
        dframe2 = dframe2.Define("HadronicW_P", "HadronicW_4Vec.P()")
        dframe2 = dframe2.Define("HadronicW_Phi", "HadronicW_4Vec.Phi()")
        dframe2 = dframe2.Define("HadronicW_CosPhi", "TMath::Cos(HadronicW_Phi)")
        dframe2 = dframe2.Define("HadronicW_Theta", "HadronicW_4Vec.Theta()")
        dframe2 = dframe2.Define("HadronicW_CosTheta", "HadronicW_4Vec.CosTheta()")
        dframe2 = dframe2.Define("HadronicW_InvariantMass", "HadronicW_4Vec.M()")
        
        # Angular separations between the two W bosons
        dframe2 = dframe2.Define("WW_DeltaR", "LeptonicW_4Vec.DeltaR(HadronicW_4Vec)")
        dframe2 = dframe2.Define("WW_DeltaPhi", "LeptonicW_4Vec.DeltaPhi(HadronicW_4Vec)")
        dframe2 = dframe2.Define("WW_DeltaTheta", "LeptonicW_Theta - HadronicW_Theta")
        dframe2 = dframe2.Define("WW_DeltaInvariantMass", "LeptonicW_InvariantMass - HadronicW_InvariantMass")
        dframe2 = dframe2.Define("WW_Angle", "LeptonicW_4Vec.Angle(HadronicW_4Vec.Vect())")
        dframe2 = dframe2.Define("WW_CosAngle", "TMath::Cos(WW_Angle)")

        # ===========================
        # Event-level invariant mass
        # ===========================
        dframe2 = dframe2.Define("Event_InvariantMass", "(Jets_4Vec[0] + Jets_4Vec[1] + MissingQuantities_4Vec[0] + IsolatedElectron_4Vec[0]).M()")

        # ===========================
        # On-shell and off-shell mass variables
        # ===========================
        dframe2 = dframe2.Define("System_OnShellMass", "TMath::Max(HadronicW_InvariantMass, LeptonicW_InvariantMass)")
        dframe2 = dframe2.Define("System_OffShellMass", "TMath::Min(HadronicW_InvariantMass, LeptonicW_InvariantMass)")
        dframe2 = dframe2.Define("System_MaxCosTheta", "TMath::Cos(TMath::Max(HadronicW_Theta, LeptonicW_Theta))")
        dframe2 = dframe2.Define("System_MinCosTheta", "TMath::Cos(TMath::Min(HadronicW_Theta, LeptonicW_Theta))")
        dframe2 = dframe2.Define("Event_EnergyImbalance", "125.0 - MissingQuantities_4Vec[0].E() - IsolatedElectron_4Vec[0].E() - Jets_4Vec[0].E() - Jets_4Vec[1].E()")

        # ===========================
        # MELA Variables (production and decay angles)
        # ===========================
        dframe2 = dframe2.Define(
            "MELA_Angles",
            "FCCAnalyses::MELA::MELACalculator::mela(Jets_4Vec[0], Jets_4Vec[1], MissingQuantities_4Vec[0], IsolatedElectron_4Vec[0], IsolatedElectron_Charge, Jet1_Charge, Jet2_Charge)",
        )
        dframe2 = dframe2.Define("MELA_Phi", "MELA_Angles.phi")
        dframe2 = dframe2.Define("MELA_CosPhi", "MELA_Angles.cosPhi")
        dframe2 = dframe2.Define("MELA_Phi1", "MELA_Angles.phi1")
        dframe2 = dframe2.Define("MELA_CosPhi1", "MELA_Angles.cosPhi1")
        dframe2 = dframe2.Define("MELA_PhiStar", "MELA_Angles.phiStar")
        dframe2 = dframe2.Define("MELA_CosPhiStar", "MELA_Angles.cosPhiStar")
        dframe2 = dframe2.Define("MELA_ThetaStar", "MELA_Angles.thetaStar")
        dframe2 = dframe2.Define("MELA_CosThetaStar", "MELA_Angles.cosThetaStar")
        dframe2 = dframe2.Define("MELA_Theta1", "MELA_Angles.theta1")
        dframe2 = dframe2.Define("MELA_CosTheta1", "MELA_Angles.cosTheta1")
        dframe2 = dframe2.Define("MELA_Theta2", "MELA_Angles.theta2")
        dframe2 = dframe2.Define("MELA_CosTheta2", "MELA_Angles.cosTheta2")

        # ===========================
        # Event-shape variables
        # ===========================
        dframe2 = dframe2.Define("EventShape_Planarity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculatePlanarity(Jet1_3Vec, Jet2_3Vec, IsolatedElectron_3Vec)")
        dframe2 = dframe2.Define("EventShape_Aplanarity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAplanarity(Jet1_3Vec, Jet2_3Vec, IsolatedElectron_3Vec)")
        dframe2 = dframe2.Define("EventShape_Sphericity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateSphericity(Jet1_3Vec, Jet2_3Vec, IsolatedElectron_3Vec)")
        dframe2 = dframe2.Define("EventShape_Asphericity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAsphericity(Jet1_3Vec, Jet2_3Vec, IsolatedElectron_3Vec)")

        # ===========================
        # Jet flavor tagging scores - Products and Logits
        # ===========================
        epsilon = 1e-10
        
        dframe2 = dframe2.Define("JetFlavor_GluonProduct", "recojet_isG[0] * recojet_isG[1]")
        dframe2 = dframe2.Define("JetFlavor_GluonProduct_Logit", f"TMath::Log((JetFlavor_GluonProduct + {epsilon}) / (1.0 - JetFlavor_GluonProduct + {epsilon}))")
        
        dframe2 = dframe2.Define("JetFlavor_UpQuarkProduct", "recojet_isU[0] * recojet_isU[1]")
        dframe2 = dframe2.Define("JetFlavor_UpQuarkProduct_Logit", f"TMath::Log((JetFlavor_UpQuarkProduct + {epsilon}) / (1.0 - JetFlavor_UpQuarkProduct + {epsilon}))")
        
        dframe2 = dframe2.Define("JetFlavor_DownQuarkProduct", "recojet_isD[0] * recojet_isD[1]")
        dframe2 = dframe2.Define("JetFlavor_DownQuarkProduct_Logit", f"TMath::Log((JetFlavor_DownQuarkProduct + {epsilon}) / (1.0 - JetFlavor_DownQuarkProduct + {epsilon}))")
        
        dframe2 = dframe2.Define("JetFlavor_StrangeQuarkProduct", "recojet_isS[0] * recojet_isS[1]")
        dframe2 = dframe2.Define("JetFlavor_StrangeQuarkProduct_Logit", f"TMath::Log((JetFlavor_StrangeQuarkProduct + {epsilon}) / (1.0 - JetFlavor_StrangeQuarkProduct + {epsilon}))")
        
        dframe2 = dframe2.Define("JetFlavor_CharmQuarkProduct", "recojet_isC[0] * recojet_isC[1]")
        dframe2 = dframe2.Define("JetFlavor_CharmQuarkProduct_Logit", f"TMath::Log((JetFlavor_CharmQuarkProduct + {epsilon}) / (1.0 - JetFlavor_CharmQuarkProduct + {epsilon}))")
        
        dframe2 = dframe2.Define("JetFlavor_BottomQuarkProduct", "recojet_isB[0] * recojet_isB[1]")
        dframe2 = dframe2.Define("JetFlavor_BottomQuarkProduct_Logit", f"TMath::Log((JetFlavor_BottomQuarkProduct + {epsilon}) / (1.0 - JetFlavor_BottomQuarkProduct + {epsilon}))")
        
        dframe2 = dframe2.Define("JetFlavor_TauProduct", "recojet_isTAU[0] * recojet_isTAU[1]")
        dframe2 = dframe2.Define("JetFlavor_TauProduct_Logit", f"TMath::Log((JetFlavor_TauProduct + {epsilon}) / (1.0 - JetFlavor_TauProduct + {epsilon}))")
        
        dframe2 = dframe2.Define("JetFlavor_UpDownQuarkProduct", "recojet_isU[0] * recojet_isD[1]")
        dframe2 = dframe2.Define("JetFlavor_UpDownQuarkProduct_Logit", f"TMath::Log((JetFlavor_UpDownQuarkProduct + {epsilon}) / (1.0 - JetFlavor_UpDownQuarkProduct + {epsilon}))")
        
        dframe2 = dframe2.Define("JetFlavor_DownUpQuarkProduct", "recojet_isD[0] * recojet_isU[1]")
        dframe2 = dframe2.Define("JetFlavor_DownUpQuarkProduct_Logit", f"TMath::Log((JetFlavor_DownUpQuarkProduct + {epsilon}) / (1.0 - JetFlavor_DownUpQuarkProduct + {epsilon}))")
        
        dframe2 = dframe2.Define("JetFlavor_StrangeCharmQuarkProduct", "recojet_isS[0] * recojet_isC[1]")
        dframe2 = dframe2.Define("JetFlavor_StrangeCharmQuarkProduct_Logit", f"TMath::Log((JetFlavor_StrangeCharmQuarkProduct + {epsilon}) / (1.0 - JetFlavor_StrangeCharmQuarkProduct + {epsilon}))")
        
        dframe2 = dframe2.Define("JetFlavor_CharmStrangeQuarkProduct", "recojet_isC[0] * recojet_isS[1]")
        dframe2 = dframe2.Define("JetFlavor_CharmStrangeQuarkProduct_Logit", f"TMath::Log((JetFlavor_CharmStrangeQuarkProduct + {epsilon}) / (1.0 - JetFlavor_CharmStrangeQuarkProduct + {epsilon}))")

        return dframe2

    # Mandatory: output function
    def output(self):
        '''
        Output variables which will be saved to output root file.
        '''
        branch_list = [
            # Photon variables
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

            # Missing quantities variables (Neutrino)
            "MissingQuantities_P",
            "MissingQuantities_E",
            "MissingQuantities_Pt",
            "MissingQuantities_Theta",
            "MissingQuantities_Phi",
            "MissingQuantities_CosTheta",
            "MissingQuantities_CosPhi",

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
            "DiJet_InvariantMass",

            # On-shell and off-shell mass variables
            "System_OnShellMass",
            "System_OffShellMass",
            "System_MaxCosTheta",
            "System_MinCosTheta",
            "Event_EnergyImbalance",

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
            "JetFlavor_GluonProduct",
            "JetFlavor_UpQuarkProduct",
            "JetFlavor_DownQuarkProduct",
            "JetFlavor_StrangeQuarkProduct",
            "JetFlavor_CharmQuarkProduct",
            "JetFlavor_BottomQuarkProduct",
            "JetFlavor_TauProduct",
            "JetFlavor_UpDownQuarkProduct",
            "JetFlavor_DownUpQuarkProduct",
            "JetFlavor_StrangeCharmQuarkProduct",
            "JetFlavor_CharmStrangeQuarkProduct",
            
            # Jet flavor tagging scores - Logits
            "JetFlavor_GluonProduct_Logit",
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
