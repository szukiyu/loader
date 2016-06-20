#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace tensorflow;

void sample_batch(std::vector<std::vector<double>> X_train , std::vector<double> y_train, int batch_size,int num_steps){

  N = X_train.size;
  len = X_train[i].size();
  //ran  = rand(N-batch_size,1)
  //ind_N = np.arange(ran,ran+batch_size)
  //ind_start = 0 //# ysuzuki added 2016/06/03                                                                                
  //#form batch                                                                                                             
  //X_batch = X_train[ind_N,ind_start:ind_start+num_steps]
  //y_batch = y_train[ind_N]
  //return X_batch,y_batch

}

std::vector<std::vector<string>> load_dataset(string filename){
  string str;
  int p;
  std::stringstream ss;                                                                                                    
  std::vector<std::vector<string>> values;
  std::ifstream file(filename);

  while(getline(file, str)){
    std::vector<string> inner;
    while( (p = str.find(",")) != string::npos ){                                                                              
      inner.push_back(str.substr(0, p));  
      str = str.substr(p+1);                                                                                                       
    }
    inner.push_back(str);                                                                                               
    values.push_back(inner);
  }                           

  return values;
}

int main(int argc, char* argv[]) {
  
  std::vector<std::vector<string>> testvalues;                                                                              
  testvalues = load_dataset("data/TEST_batch1000");
  std::vector<std::vector<string>> trainvalues;                                                                              
  trainvalues = load_dataset("data/TEST_batch1000");

  std::cout<<"aa"<<std::endl;

  std::stringstream ss;
  double v;

  std::vector<std::vector<double>> X_train_norm, X_test_norm;
  std::vector<double> y_train_norm, y_test_norm;

  for(unsigned int i = 0; i < trainvalues.size(); ++i){
    for(unsigned int j = 0; j < trainvalues[i].size(); ++j){

      ss << trainvalues[i][j];
      ss >> v;

      if(j != 0)
	X_train_norm[i].push_back(v);
      else
	y_train_norm.push_back(v);

      //std::cout<<ss<<std::endl;
    }
  }

  for(unsigned int i = 0; i < testvalues.size(); ++i){
    for(unsigned int j = 0; j < testvalues[i].size(); ++j){

      ss << testvalues[i][j];
      ss >> v;

      if(j != 0)
	X_test_norm[i].push_back(v);
      else
	y_test_norm.push_back(v);
    }
  }



  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), "models/graph.pb", &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Setup inputs and outputs:

  // Our graph doesn't require any inputs, since it specifies default values,
  // but we'll change an input to demonstrate.
  Tensor a(DT_FLOAT, TensorShape());
  a.scalar<float>()() = 3.0;

  Tensor b(DT_FLOAT, TensorShape());
  b.scalar<float>()() = 2.0;

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
    { "a", a },
    { "b", b },
  };

  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  // Run the session, evaluating our "c" operation from the graph
  status = session->Run(inputs, {"c"}, {}, &outputs);
  //status = session->Run(input_data: X_batch,targets:y_batch, initial_state:state,keep_prob:1});
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Grab the first output (we only evaluated one graph node: "c")
  // and convert the node to a scalar representation.
  auto output_c = outputs[0].scalar<float>();

  // (There are similar methods for vectors and matrices here:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

  // Print the results
  std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
  std::cout << output_c() << "\n"; // 30

  // Free any resources used by the session
  session->Close();
  return 0;
}
