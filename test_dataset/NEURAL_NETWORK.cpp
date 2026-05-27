NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size, double lr)
{
    learning_rate = lr;
    weights_input_hidden = MatrixXd::Random(hidden_size, input_size);
    weights_hidden_output = MatrixXd::Random(output_size, hidden_size);
    bias_hidden = VectorXd::Random(hidden_size);
    bias_output = VectorXd::Random(output_size);
}
VectorXd NeuralNetwork::feedforward(const VectorXd& input){
    VectorXd hidden_input = weights_input_hidden * input + bias_hidden;
    VectorXd hidden_output = sigmoid(hidden_input);
    VectorXd final_input = weights_hidden_output * hidden_output + bias_output;
    VectorXd final_output = sigmoid(final_input);
    return final_output;
    
}