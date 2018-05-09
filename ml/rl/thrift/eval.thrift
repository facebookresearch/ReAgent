namespace py ml.rl.thrift.eval

struct ValueInputModelParameters {
  1: string name,
  2: optional string path,
}

struct PolicyEvaluatorParameters {
  1: list<ValueInputModelParameters> value_input_models,
  2: string propensity_net_path,
  3: string db_type,
  4: map<string, double> global_value_inputs,
}
