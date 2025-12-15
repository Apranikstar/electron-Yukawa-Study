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
        #self.output_dir = 'electron_yukawa/stage1'

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
        # Combined masses
        # ===========================
        dframe2 = dframe2.Define("ElectronNeutrino_InvariantMass", "(MissingQuantities_4Vec[0] + IsolatedElectron_4Vec[0]).M()")

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
        # Angular separations: Jet1 vs Jet2
        # ===========================
        dframe2 = dframe2.Define("Jet1Jet2_DeltaR", "Jets_4Vec[0].DeltaR(Jets_4Vec[1])")
        dframe2 = dframe2.Define("Jet1Jet2_DeltaPhi", "Jets_4Vec[0].DeltaPhi(Jets_4Vec[1])")
        dframe2 = dframe2.Define("Jet1Jet2_DeltaTheta", "Jet1_Theta - Jet2_Theta")
        dframe2 = dframe2.Define("Jet1Jet2_Angle", "Jets_4Vec[0].Angle(Jets_4Vec[1].Vect())")
        dframe2 = dframe2.Define("Jet1Jet2_CosAngle", "TMath::Cos(Jet1Jet2_Angle)")

        # ===========================
        # Angular separations: Electron vs Jets
        # ===========================
        dframe2 = dframe2.Define("ElectronJet1_DeltaR", "IsolatedElectron_4Vec[0].DeltaR(Jets_4Vec[0])")
        dframe2 = dframe2.Define("ElectronJet1_DeltaPhi", "IsolatedElectron_4Vec[0].DeltaPhi(Jets_4Vec[0])")
        dframe2 = dframe2.Define("ElectronJet1_Angle", "IsolatedElectron_4Vec[0].Angle(Jets_4Vec[0].Vect())")
        dframe2 = dframe2.Define("ElectronJet1_CosAngle", "TMath::Cos(ElectronJet1_Angle)")

        dframe2 = dframe2.Define("ElectronJet2_DeltaR", "IsolatedElectron_4Vec[0].DeltaR(Jets_4Vec[1])")
        dframe2 = dframe2.Define("ElectronJet2_DeltaPhi", "IsolatedElectron_4Vec[0].DeltaPhi(Jets_4Vec[1])")
        dframe2 = dframe2.Define("ElectronJet2_Angle", "IsolatedElectron_4Vec[0].Angle(Jets_4Vec[1].Vect())")
        dframe2 = dframe2.Define("ElectronJet2_CosAngle", "TMath::Cos(ElectronJet2_Angle)")

        dframe2 = dframe2.Define("ElectronJets_MaxDeltaR", "TMath::Max(ElectronJet1_DeltaR, ElectronJet2_DeltaR)")
        dframe2 = dframe2.Define("ElectronJets_MinDeltaR", "TMath::Min(ElectronJet1_DeltaR, ElectronJet2_DeltaR)")
        dframe2 = dframe2.Define("ElectronJets_MaxDeltaPhi", "TMath::Max(ElectronJet1_DeltaPhi, ElectronJet2_DeltaPhi)")
        dframe2 = dframe2.Define("ElectronJets_MinDeltaPhi", "TMath::Min(ElectronJet1_DeltaPhi, ElectronJet2_DeltaPhi)")
        dframe2 = dframe2.Define("ElectronJets_MaxCosAngle", "TMath::Max(ElectronJet1_CosAngle, ElectronJet2_CosAngle)")
        dframe2 = dframe2.Define("ElectronJets_MinCosAngle", "TMath::Min(ElectronJet1_CosAngle, ElectronJet2_CosAngle)")

        # ===========================
        # Angular separations: MissingQuantities vs Electron
        # ===========================
        dframe2 = dframe2.Define("MissingQuantitiesElectron_DeltaR", "MissingQuantities_4Vec[0].DeltaR(IsolatedElectron_4Vec[0])")
        dframe2 = dframe2.Define("MissingQuantitiesElectron_DeltaPhi", "MissingQuantities_4Vec[0].DeltaPhi(IsolatedElectron_4Vec[0])")
        dframe2 = dframe2.Define("MissingQuantitiesElectron_Angle", "MissingQuantities_4Vec[0].Angle(IsolatedElectron_4Vec[0].Vect())")
        dframe2 = dframe2.Define("MissingQuantitiesElectron_CosAngle", "TMath::Cos(MissingQuantitiesElectron_Angle)")

        # ===========================
        # Angular separations: MissingQuantities vs Jets
        # ===========================
        dframe2 = dframe2.Define("MissingQuantitiesJet1_DeltaR", "MissingQuantities_4Vec[0].DeltaR(Jets_4Vec[0])")
        dframe2 = dframe2.Define("MissingQuantitiesJet1_DeltaPhi", "MissingQuantities_4Vec[0].DeltaPhi(Jets_4Vec[0])")
        dframe2 = dframe2.Define("MissingQuantitiesJet1_Angle", "MissingQuantities_4Vec[0].Angle(Jets_4Vec[0].Vect())")
        dframe2 = dframe2.Define("MissingQuantitiesJet1_CosAngle", "TMath::Cos(MissingQuantitiesJet1_Angle)")

        dframe2 = dframe2.Define("MissingQuantitiesJet2_DeltaR", "MissingQuantities_4Vec[0].DeltaR(Jets_4Vec[1])")
        dframe2 = dframe2.Define("MissingQuantitiesJet2_DeltaPhi", "MissingQuantities_4Vec[0].DeltaPhi(Jets_4Vec[1])")
        dframe2 = dframe2.Define("MissingQuantitiesJet2_Angle", "MissingQuantities_4Vec[0].Angle(Jets_4Vec[1].Vect())")
        dframe2 = dframe2.Define("MissingQuantitiesJet2_CosAngle", "TMath::Cos(MissingQuantitiesJet2_Angle)")

        dframe2 = dframe2.Define("MissingQuantitiesJets_MaxDeltaR", "TMath::Max(MissingQuantitiesJet1_DeltaR, MissingQuantitiesJet2_DeltaR)")
        dframe2 = dframe2.Define("MissingQuantitiesJets_MinDeltaR", "TMath::Min(MissingQuantitiesJet1_DeltaR, MissingQuantitiesJet2_DeltaR)")
        dframe2 = dframe2.Define("MissingQuantitiesJets_MaxDeltaPhi", "TMath::Max(MissingQuantitiesJet1_DeltaPhi, MissingQuantitiesJet2_DeltaPhi)")
        dframe2 = dframe2.Define("MissingQuantitiesJets_MinDeltaPhi", "TMath::Min(MissingQuantitiesJet1_DeltaPhi, MissingQuantitiesJet2_DeltaPhi)")
        dframe2 = dframe2.Define("MissingQuantitiesJets_MaxCosAngle", "TMath::Max(MissingQuantitiesJet1_CosAngle, MissingQuantitiesJet2_CosAngle)")
        dframe2 = dframe2.Define("MissingQuantitiesJets_MinCosAngle", "TMath::Min(MissingQuantitiesJet1_CosAngle, MissingQuantitiesJet2_CosAngle)")

        # ===========================
        # Event-level invariant mass
        # ===========================
        dframe2 = dframe2.Define("Event_InvariantMass", "(Jets_4Vec[0] + Jets_4Vec[1] + MissingQuantities_4Vec[0] + IsolatedElectron_4Vec[0]).M()")

        # ===========================
        # Combined system masses
        # ===========================
        dframe2 = dframe2.Define("ElectronDiJet_InvariantMass", "(Jets_4Vec[0] + Jets_4Vec[1] + IsolatedElectron_4Vec[0]).M()")
        dframe2 = dframe2.Define("ElectronJet1_InvariantMass", "(Jets_4Vec[0] + IsolatedElectron_4Vec[0]).M()")
        dframe2 = dframe2.Define("ElectronJet2_InvariantMass", "(Jets_4Vec[1] + IsolatedElectron_4Vec[0]).M()")
        dframe2 = dframe2.Define("DiJet_Energy", "(Jets_4Vec[0] + Jets_4Vec[1]).E()")

        # ===========================
        # System angular properties
        # ===========================
        dframe2 = dframe2.Define("ElectronDiJet_Phi", "(Jets_4Vec[0] + Jets_4Vec[1] + IsolatedElectron_4Vec[0]).Phi()")
        dframe2 = dframe2.Define("DiJet_Phi", "(Jets_4Vec[0] + Jets_4Vec[1]).Phi()")

        # ===========================
        # W boson (leptonic) properties
        # ===========================
        dframe2 = dframe2.Define("WBosonLeptonic_InvariantMass", "(IsolatedElectron_4Vec[0] + MissingQuantities_4Vec[0]).M()")
        dframe2 = dframe2.Define("WBosonLeptonic_Theta", "(IsolatedElectron_4Vec[0] + MissingQuantities_4Vec[0]).Theta()")

        # ===========================
        # On-shell and off-shell mass variables
        # ===========================
        dframe2 = dframe2.Define("System_OnShellMass", "TMath::Max((Jets_4Vec[0]+Jets_4Vec[1]).M(), WBosonLeptonic_InvariantMass)")
        dframe2 = dframe2.Define("System_OffShellMass", "TMath::Min((Jets_4Vec[0]+Jets_4Vec[1]).M(), WBosonLeptonic_InvariantMass)")
        dframe2 = dframe2.Define("System_MaxCosTheta", "TMath::Cos(TMath::Max((Jets_4Vec[0]+Jets_4Vec[1]).Theta(), WBosonLeptonic_Theta))")
        dframe2 = dframe2.Define("System_MinCosTheta", "TMath::Cos(TMath::Min((Jets_4Vec[0]+Jets_4Vec[1]).Theta(), WBosonLeptonic_Theta))")
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
        # Jet flavor tagging scores - Products
        # ===========================
        dframe2 = dframe2.Define("JetFlavor_GluonProduct", "recojet_isG[0] * recojet_isG[1]")
        dframe2 = dframe2.Define("JetFlavor_UpQuarkProduct", "recojet_isU[0] * recojet_isU[1]")
        dframe2 = dframe2.Define("JetFlavor_DownQuarkProduct", "recojet_isD[0] * recojet_isD[1]")
        dframe2 = dframe2.Define("JetFlavor_StrangeQuarkProduct", "recojet_isS[0] * recojet_isS[1]")
        dframe2 = dframe2.Define("JetFlavor_CharmQuarkProduct", "recojet_isC[0] * recojet_isC[1]")
        dframe2 = dframe2.Define("JetFlavor_BottomQuarkProduct", "recojet_isB[0] * recojet_isB[1]")
        dframe2 = dframe2.Define("JetFlavor_TauProduct", "recojet_isTAU[0] * recojet_isTAU[1]")


        dframe2 = dframe2.Define("JetFlavor_UpDownQuarkProduct", "recojet_isU[0] * recojet_isD[1]")
        dframe2 = dframe2.Define("JetFlavor_DownUpQuarkProduct", "recojet_isD[0] * recojet_isU[1]")

        dframe2 = dframe2.Define("JetFlavor_StrangeCharmQuarkProduct", "recojet_isS[0] * recojet_isC[1]")
        dframe2 = dframe2.Define("JetFlavor_CharmStrangeQuarkProduct", "recojet_isC[0] * recojet_isS[1]") 

       

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

            # Missing quantities variables
            "MissingQuantities_P",
            "MissingQuantities_E",
            "MissingQuantities_Pt",
            "MissingQuantities_Theta",
            "MissingQuantities_Phi",
            "MissingQuantities_CosTheta",
            "MissingQuantities_CosPhi",

            # Combined system masses
            "ElectronNeutrino_InvariantMass",
            "DiJet_InvariantMass",

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

            # Angular separations: Jet1 vs Jet2
            "Jet1Jet2_DeltaR",
            "Jet1Jet2_DeltaPhi",
            "Jet1Jet2_DeltaTheta",
            "Jet1Jet2_Angle",
            "Jet1Jet2_CosAngle",

            # Angular separations: Electron vs Jets
            "ElectronJet1_DeltaR",
            "ElectronJet1_DeltaPhi",
            "ElectronJet1_Angle",
            "ElectronJet1_CosAngle",
            "ElectronJet2_DeltaR",
            "ElectronJet2_DeltaPhi",
            "ElectronJet2_Angle",
            "ElectronJet2_CosAngle",
            "ElectronJets_MaxDeltaR",
            "ElectronJets_MinDeltaR",
            "ElectronJets_MaxDeltaPhi",
            "ElectronJets_MinDeltaPhi",
            "ElectronJets_MaxCosAngle",
            "ElectronJets_MinCosAngle",

            # Angular separations: MissingQuantities vs Electron
            "MissingQuantitiesElectron_DeltaR",
            "MissingQuantitiesElectron_DeltaPhi",
            "MissingQuantitiesElectron_Angle",
            "MissingQuantitiesElectron_CosAngle",

            # Angular separations: MissingQuantities vs Jets
            "MissingQuantitiesJet1_DeltaR",
            "MissingQuantitiesJet1_DeltaPhi",
            "MissingQuantitiesJet1_Angle",
            "MissingQuantitiesJet1_CosAngle",
            "MissingQuantitiesJet2_DeltaR",
            "MissingQuantitiesJet2_DeltaPhi",
            "MissingQuantitiesJet2_Angle",
            "MissingQuantitiesJet2_CosAngle",
            "MissingQuantitiesJets_MaxDeltaR",
            "MissingQuantitiesJets_MinDeltaR",
            "MissingQuantitiesJets_MaxDeltaPhi",
            "MissingQuantitiesJets_MinDeltaPhi",
            "MissingQuantitiesJets_MaxCosAngle",
            "MissingQuantitiesJets_MinCosAngle",

            # Event-level masses
            "Event_InvariantMass",
            "ElectronDiJet_InvariantMass",
            "ElectronJet1_InvariantMass",
            "ElectronJet2_InvariantMass",
            "DiJet_Energy",

            # System angular properties
            "ElectronDiJet_Phi",
            "DiJet_Phi",

            # W boson properties
            "WBosonLeptonic_InvariantMass",
            "WBosonLeptonic_Theta",
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

            # Jet flavor tagging scores
            "JetFlavor_GluonProduct",
            "JetFlavor_UpQuarkProduct",
            "JetFlavor_DownQuarkProduct",
            "JetFlavor_StrangeQuarkProduct",
            "JetFlavor_CharmQuarkProduct",
            "JetFlavor_BottomQuarkProduct",
            "JetFlavor_TauProduct",
            "JetFlavor_UpDownQuarkProduct",
            "JetFlavor_DownUpQuarkProduct" ,
            "JetFlavor_StrangeCharmQuarkProduct" ,
            "JetFlavor_CharmStrangeQuarkProduct",
        ]
        return branch_list
