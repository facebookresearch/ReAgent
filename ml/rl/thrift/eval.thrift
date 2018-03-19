namespace py ml.rl.thrift.eval

struct ValueModelParameters {
  1: string name,
  2: optional string path,
}

struct PolicyEvaluatorParameters {
  1: list<ValueModelParameters> value_models,
  2: string propensity_net_path,
  3: string db_type,
}
