import collections
import math
import itertools 

# --- 0. Variable Definitions ---
VARIABLES = [
    "IsSummer", "HasFlu", "HasFoodPoisoning", "HasHayFever", "HasPneumonia",
    "HasRespiratoryProblems", "HasGastricProblems", "HasRash", "Coughs",
    "IsFatigued", "Vomits", "HasFever"
]
NUM_VARS = len(VARIABLES)
VAR_TO_IDX = {name: i for i, name in enumerate(VARIABLES)}

# --- Helper Functions ---
def int_to_assignment(val, num_vars):
    """Converts an integer to a tuple of boolean assignments (LSB is var 0)."""
    return tuple(bool((val >> i) & 1) for i in range(num_vars))

def assignment_to_int(assignment_tuple):
    """Converts a tuple of boolean assignments to an integer (LSB is var 0)."""
    val = 0
    for i, bit in enumerate(assignment_tuple):
        if bit:
            val |= (1 << i)
    return val

def load_joint_dat(filepath="joint.dat"):
    """Loads the true joint probability distribution."""
    true_jpd = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    integer_val, prob = int(parts[0]), float(parts[1])
                    true_jpd[integer_val] = prob
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure it's in the same directory.")
        exit()
    if not true_jpd:
        print(f"Error: {filepath} was found but no data was loaded. Check file format.")
        exit()
    return true_jpd

def load_dataset_dat(filepath="dataset.dat"):
    """Loads the dataset samples."""
    dataset = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                dataset.append(int(line.strip()))
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please ensure it's in the same directory.")
        exit()
    if not dataset:
        print(f"Error: {filepath} was found but no data was loaded. Check file format.")
        exit()
    return dataset

class BayesianNetwork:
    def __init__(self, structure, variables, var_to_idx):
        """
        structure: dict, e.g., {'VarName': ['Parent1', 'Parent2']}
        variables: ordered list of variable names
        var_to_idx: mapping from variable name to index
        """
        self.variables = variables
        self.var_to_idx = var_to_idx
        self.structure = structure
        self.cpts = {} 
        for var_name in self.variables:
            if var_name not in self.structure:
                self.structure[var_name] = [] 
    def get_num_parameters(self):
        """Calculates the number of parameters required by the model's CPTs."""
        num_params = 0
        for var_name in self.variables:
            parents = self.structure.get(var_name, [])
            num_parent_configs = 2**len(parents)
           
            num_params += num_parent_configs
        return num_params

    def estimate_parameters(self, dataset_int_assignments):
        """
        Estimates CPTs from the dataset using maximum likelihood estimation (counts and normalization).
        """
        print("Estimating parameters...")
        self.cpts = {}
        dataset_assignments = [int_to_assignment(val, len(self.variables)) for val in dataset_int_assignments]

        for var_idx, var_name in enumerate(self.variables):
            parents_names = self.structure.get(var_name, [])
            parent_indices = [self.var_to_idx[p_name] for p_name in parents_names]
            self.cpts[var_name] = {}

            num_parents = len(parents_names)
            parent_value_combinations = list(itertools.product([False, True], repeat=num_parents))

            for parent_vals_tuple in parent_value_combinations:
                count_parent_config = 0
                count_var_true_and_parent_config = 0

                for sample_assignment in dataset_assignments:
                    current_sample_parent_values = tuple(sample_assignment[p_idx] for p_idx in parent_indices)
                    if current_sample_parent_values == parent_vals_tuple:
                        count_parent_config += 1
                        if sample_assignment[var_idx]: # If var_name is true
                            count_var_true_and_parent_config += 1
                
                if count_parent_config > 0:
                    prob_true = count_var_true_and_parent_config / count_parent_config
                else:
                    
                    prob_true = 0.5 
                self.cpts[var_name][parent_vals_tuple] = prob_true
        print("Parameters estimated.")

    def get_prob_of_full_assignment(self, full_assignment_tuple):
        """
        Calculates the probability of a full assignment using the CPTs.
        full_assignment_tuple: A tuple of booleans representing the state of all variables.
        """
        if not self.cpts:
            raise ValueError("CPTs have not been estimated. Call estimate_parameters first.")
        
        joint_prob = 1.0
        for var_idx, var_name in enumerate(self.variables):
            var_value = full_assignment_tuple[var_idx]
            parents_names = self.structure.get(var_name, [])
            parent_indices = [self.var_to_idx[p_name] for p_name in parents_names]
            
            current_parent_values_tuple = tuple(full_assignment_tuple[p_idx] for p_idx in parent_indices)
            
            prob_var_true_given_parents = self.cpts[var_name][current_parent_values_tuple]
            
            if var_value: # Variable is True
                joint_prob *= prob_var_true_given_parents
            else: # Variable is False
                joint_prob *= (1.0 - prob_var_true_given_parents)
        return joint_prob

    def calculate_l1_distance(self, true_jpd):
        """Calculates L1 distance between model's JPD and the true JPD."""
        print("Calculating L1 distance...")
        l1_distance = 0.0
        num_total_assignments = 2**len(self.variables)
        for i in range(num_total_assignments):
            assignment_tuple = int_to_assignment(i, len(self.variables))
            prob_model = self.get_prob_of_full_assignment(assignment_tuple)
            prob_true = true_jpd.get(i, 0.0)
            l1_distance += abs(prob_model - prob_true)
        print(f"L1 Distance: {l1_distance}")
        return l1_distance

    def calculate_kl_divergence(self, true_jpd):
        """Calculates KL divergence D_KL(P_true || P_model)."""
        print("Calculating KL divergence (True || Model)...")
        kl_div = 0.0
        epsilon = 1e-12 # To avoid log(0) or division by zero
        num_total_assignments = 2**len(self.variables)

        for i in range(num_total_assignments):
            assignment_tuple = int_to_assignment(i, len(self.variables))
            p_true = true_jpd.get(i, 0.0)
            
            if p_true < epsilon:
                continue # Term is 0 if p_true is 0

            p_model = self.get_prob_of_full_assignment(assignment_tuple)
            
            if p_model < epsilon:
              
                p_model = epsilon 
            
            kl_div += p_true * math.log(p_true / p_model)
        print(f"KL Divergence (True || Model): {kl_div}")
        return kl_div

    def query(self, query_vars_dict, evidence_vars_dict, use_true_jpd_for_calc=None):
        """
        Performs exact inference by enumeration.
        query_vars_dict: {'VarName': True/False} for the state of query variables.
        evidence_vars_dict: {'VarName': True/False} for observed evidence.
        use_true_jpd_for_calc: If not None, uses this true JPD dict instead of model CPTs.
        Returns P(QueryVars | EvidenceVars).
        """
        numerator_sum = 0.0
        denominator_sum = 0.0
        num_total_assignments = 2**len(self.variables)

        for i in range(num_total_assignments):
            full_assignment_tuple = int_to_assignment(i, len(self.variables))
            
            prob_full_assignment = 0.0
            if use_true_jpd_for_calc:
                prob_full_assignment = use_true_jpd_for_calc.get(i, 0.0)
            else:
                if not self.cpts: # Ensure model is trained if not using true JPD
                    raise ValueError("Model CPTs not estimated. Call estimate_parameters or provide true_jpd.")
                prob_full_assignment = self.get_prob_of_full_assignment(full_assignment_tuple)

            if prob_full_assignment < 1e-12 and not use_true_jpd_for_calc:
                continue


            
            consistent_with_evidence = True
            for var_name, var_value in evidence_vars_dict.items():
                var_idx = self.var_to_idx[var_name]
                if full_assignment_tuple[var_idx] != var_value:
                    consistent_with_evidence = False
                    break
            
            if consistent_with_evidence:
                denominator_sum += prob_full_assignment
                
                consistent_with_query = True
                for var_name, var_value in query_vars_dict.items():
                    var_idx = self.var_to_idx[var_name]
                    if full_assignment_tuple[var_idx] != var_value:
                        consistent_with_query = False
                        break
                
                if consistent_with_query:
                    numerator_sum += prob_full_assignment
        
        if denominator_sum < 1e-12: 
            return 0.0 
        return numerator_sum / denominator_sum

    def query_distribution(self, query_var_name, evidence_vars_dict, use_true_jpd_for_calc=None):
        """ Helper for getting P(QueryVar=True|E) and P(QueryVar=False|E) """
        prob_true = self.query({query_var_name: True}, evidence_vars_dict, use_true_jpd_for_calc)
       
        prob_false = self.query({query_var_name: False}, evidence_vars_dict, use_true_jpd_for_calc)
        
        total_prob = prob_true + prob_false
        if total_prob < 1e-9: 
            return {True: 0.0, False: 0.0}
      
        return {True: prob_true / total_prob, False: prob_false / total_prob}


# --- Main Execution ---
if __name__ == "__main__":


    intuitive_model_structure = {
        "IsSummer": [],  

        "HasFlu": ["IsSummer"],          
        "HasFoodPoisoning": [],    
        "HasHayFever": ["IsSummer"],     

        "HasPneumonia": ["HasFlu"],     

       
        "HasRespiratoryProblems": ["HasFlu", "HasHayFever", "HasPneumonia"],
        "HasGastricProblems": ["HasFoodPoisoning"],

        
        "Coughs": ["HasRespiratoryProblems"],
        "Vomits": ["HasGastricProblems"],
        "HasFever": ["HasFlu", "HasPneumonia", "HasFoodPoisoning"], 
        "IsFatigued": ["HasFlu", "HasPneumonia", "HasFoodPoisoning", "HasHayFever", "HasFever"], 
        "HasRash": ["HasHayFever"]      
                                         
    }

    bn = BayesianNetwork(structure=intuitive_model_structure, variables=VARIABLES, var_to_idx=VAR_TO_IDX)
    num_model_params = bn.get_num_parameters()
    print(f"Model structure defined with {num_model_params} parameters.")
    print(f"A full joint distribution would require {2**NUM_VARS - 1} parameters.")

    # --- Load Data ---
    print("\nLoading data...")
    true_jpd = load_joint_dat()
    dataset_samples = load_dataset_dat()
    print(f"Loaded true JPD ({len(true_jpd)} entries) and dataset ({len(dataset_samples)} samples).")

    # --- 2. Estimate Parameters ---
    bn.estimate_parameters(dataset_samples)

    # --- 3. Model Accuracy ---
    print("\n--- Model Accuracy ---")
    l1_dist = bn.calculate_l1_distance(true_jpd)
    kl_div = bn.calculate_kl_divergence(true_jpd)

    # --- 4. Querying ---
    print("\n--- Querying Examples ---")

    # Query 1 (from assignment): P(HasFlu | Coughs=true, HasFever=true)
    q1_evidence = {"Coughs": True, "HasFever": True}
    q1_query_var = "HasFlu"
    print(f"\nQuery 1: P({q1_query_var}=True | Coughs=T, HasFever=T)")
    prob_flu_model_q1 = bn.query({q1_query_var: True}, q1_evidence)
    print(f"  Model Result: {prob_flu_model_q1:.4f}")
    prob_flu_true_q1 = bn.query({q1_query_var: True}, q1_evidence, use_true_jpd_for_calc=true_jpd)
    print(f"  True JPD Result: {prob_flu_true_q1:.4f}")

    # Query 2 (from assignment, simplified to one symptom): P(Coughs | HasPneumonia=true)
    # The full query asks for a joint distribution over 5 symptoms, which is 32 states.
    # We'll show one symptom distribution for brevity in console output.
    q2_evidence = {"HasPneumonia": True}
    q2_query_symptom = "Coughs" # Example symptom
    print(f"\nQuery 2 (Distribution for one symptom): P({q2_query_symptom} | HasPneumonia=T)")
    dist_model_q2 = bn.query_distribution(q2_query_symptom, q2_evidence)
    print(f"  Model: P({q2_query_symptom}=T|E)={dist_model_q2[True]:.4f}, P({q2_query_symptom}=F|E)={dist_model_q2[False]:.4f}")
    dist_true_q2 = bn.query_distribution(q2_query_symptom, q2_evidence, use_true_jpd_for_calc=true_jpd)
    print(f"  True JPD: P({q2_query_symptom}=T|E)={dist_true_q2[True]:.4f}, P({q2_query_symptom}=F|E)={dist_true_q2[False]:.4f}")

    # Query 3 (from assignment): P(Vomits | IsSummer=true)
    q3_evidence = {"IsSummer": True}
    q3_query_var = "Vomits"
    print(f"\nQuery 3: P({q3_query_var}=True | IsSummer=T)")
    prob_vomits_model_q3 = bn.query({q3_query_var: True}, q3_evidence)
    print(f"  Model Result: {prob_vomits_model_q3:.4f}")
    prob_vomits_true_q3 = bn.query({q3_query_var: True}, q3_evidence, use_true_jpd_for_calc=true_jpd)
    print(f"  True JPD Result: {prob_vomits_true_q3:.4f}")

    # --- Custom Queries (Demonstrating different reasoning types) ---
    print("\n--- Custom Queries ---")

    # Causal Reasoning: P(Coughs=T | HasFlu=T)
    # Does having the flu increase the chance of coughing according to the model?
    cq_causal_evidence = {"HasFlu": True}
    cq_causal_query = "Coughs"
    print(f"\nCustom Causal Query: P({cq_causal_query}=True | HasFlu=T)")
    prob_model_cq_causal = bn.query({cq_causal_query: True}, cq_causal_evidence)
    print(f"  Model Result: {prob_model_cq_causal:.4f}")
    prob_true_cq_causal = bn.query({cq_causal_query: True}, cq_causal_evidence, use_true_jpd_for_calc=true_jpd)
    print(f"  True JPD Result: {prob_true_cq_causal:.4f}")

    # Evidential Reasoning: P(HasFoodPoisoning=T | Vomits=T)
    # If a patient is vomiting, how likely is it food poisoning?
    cq_evidential_evidence = {"Vomits": True}
    cq_evidential_query = "HasFoodPoisoning"
    print(f"\nCustom Evidential Query: P({cq_evidential_query}=True | Vomits=T)")
    prob_model_cq_evidential = bn.query({cq_evidential_query: True}, cq_evidential_evidence)
    print(f"  Model Result: {prob_model_cq_evidential:.4f}")
    prob_true_cq_evidential = bn.query({cq_evidential_query: True}, cq_evidential_evidence, use_true_jpd_for_calc=true_jpd)
    print(f"  True JPD Result: {prob_true_cq_evidential:.4f}")

    # Inter-causal Reasoning (Explaining Away):
    # P(HasFlu=T | Coughs=T) vs P(HasFlu=T | Coughs=T, HasHayFever=T)
    # If we know the patient has hay fever (another cause of coughs), does it make flu less likely as the cause of the cough?
    inter_q_query = "HasFlu"
    inter_q_ev1 = {"Coughs": True} # Evidence: just coughs
    inter_q_ev2 = {"Coughs": True, "HasHayFever": True} # Evidence: coughs AND hay fever

    print(f"\nCustom Inter-causal Query (Explaining Away for {inter_q_query}):")
    prob_flu_cough_model = bn.query({inter_q_query: True}, inter_q_ev1)
    prob_flu_cough_true = bn.query({inter_q_query: True}, inter_q_ev1, use_true_jpd_for_calc=true_jpd)
    print(f"  P({inter_q_query}=T | Coughs=T): Model={prob_flu_cough_model:.4f}, True={prob_flu_cough_true:.4f}")
    
    prob_flu_cough_hayfever_model = bn.query({inter_q_query: True}, inter_q_ev2)
    prob_flu_cough_hayfever_true = bn.query({inter_q_query: True}, inter_q_ev2, use_true_jpd_for_calc=true_jpd)
    print(f"  P({inter_q_query}=T | Coughs=T, HasHayFever=T): Model={prob_flu_cough_hayfever_model:.4f}, True={prob_flu_cough_hayfever_true:.4f}")
    
    if prob_flu_cough_hayfever_model < prob_flu_cough_model:
        print("  Model shows 'explaining away' effect: P(Flu|Cough,HayFever) < P(Flu|Cough).")
    if prob_flu_cough_hayfever_true < prob_flu_cough_true:
        print("  True JPD shows 'explaining away' effect.")

    print("\n--- End of Script ---")
