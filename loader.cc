#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace tensorflow;

std::vector<std::vector<double>> sample_batch(std::vector<std::vector<double>> X_train , std::vector<double> y_train, int batch_size,int num_steps){
  int N = X_train.size();
  //int len = X_train[0].size();
  std::vector<std::vector<double>> X_batch(batch_size);
  int ran  =  (int)(rand()*(N-batch_size+1.0)/(1.0+RAND_MAX));
  int ind_N[batch_size];
  
  for(int i=0; i < batch_size; ++i){
    X_batch[i].push_back(X_train[ran+i][0]);
    for(int j=0; j < num_steps; ++j){
      X_batch[i].push_back(X_train[ran+i][j+1]);
    } 
  }

  return X_batch;

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

  std::stringstream ss;
  double v;

  std::vector<std::vector<double>> X_train_norm(1000), X_test_norm(1000);
  std::vector<double> y_train_norm(1000), y_test_norm(1000);

  for(unsigned int i = 0; i < trainvalues.size(); ++i){
    for(unsigned int j = 0; j < trainvalues[i].size(); ++j){

      ss << trainvalues[i][j];
      ss >> v;

      if(j != 0)
	X_train_norm[i].push_back(v);
      else
	y_train_norm.push_back(v);

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

  std::vector<std::vector<double>> batchx = sample_batch(X_test_norm, y_test_norm, 32, 30);
  for(unsigned int i = 0; i < batchx.size(); ++i){
    for(unsigned int j = 0; j < batchx[i].size(); ++j){
      //std::cout << batchx[i][j];
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
  //Tensor a(DT_FLOAT, TensorShape());

  /*
  Tensor a_temp;
  // Initialize `a_temp`...

  const auto& a_tensor = a_temp.shaped<float, 3>(
    {4, 4, 4});

  Eigen::array<ptrdiff_t, 3> patch_dims;
  patch_dims[0] = 1;
  patch_dims[1] = 2;
  patch_dims[2] = 3;

  const auto& patch_expr = a_tensor.extract_patches(patch_dims);
  std::cout<<"ee"<<std::endl;
  Eigen::Tensor<float, 4, Eigen::RowMajor> patch = patch_expr.eval();
  std::cout<<"patch="<<patch<<std::endl;
  */

  Tensor a(DT_FLOAT, TensorShape({32,30}));
  //a.scalar<float>()() = 3.0;
 a.flat_inner_dims<float>().setRandom();
  //std::cout<<"a.matrix<T>() is "<< a.shaped<double, 2>({4, 15})<<std::endl;

 Tensor b(DT_FLOAT, TensorShape({30}));
 b.flat<float>().setRandom();
 //std::cout<<"b.matrix<double>() is "<< b.matrix<double>()<<std::endl;

 Tensor c(DT_FLOAT, TensorShape());
 c.scalar<float>()() = 0.5;

 std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
   { "input_data", a },
   { "Targets", b },
   { "Drop_out_keep_prob", c },
 };
  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  //auto operationStatus = session->Run(inputs, {"init_all_vars_op2"}, {}, &outputs);
  //status = session->Run(inputs, {}, {"init_all_vars_op2"}, &outputs);
  Run the session, evaluating our "c" operation from the graph
  status = session->Run(inputs, {"Softmax/costvalue"}, {}, &outputs);
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
  //std::cout << outputs[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
  std::cout << output_c() << "\n"; // 30

  // Free any resources used by the session
  session->Close();
  return 0;
}
