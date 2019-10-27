void web_client_init(char *addr);
void web_client_clean();
int web_send_inputs_to_net(int net_id, double *inputs, int len_inputs,
<------><------>double *outputs, int len_outputs, char *error);
int web_send_train_to_net(int net_id, double *inputs, int len_inputs,
<------><------>double *outputs, int len_outputs, char *error);