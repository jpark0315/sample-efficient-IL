
def test_model(states, agent):
    loss_pred = []
    for i in range(20):
        agent_action = agent.select_action(states, i)
        model_n_obs, info = model.predict_next_states(states, agent_action, deterministic = True)
        model_loss = model.validate(states, agent_action, model_n_obs, verbose = False)
        states = model_n_obs
        
        print(np.asarray(model_loss).mean())
        loss_pred.append(np.asarray(model_loss).mean(), )
    return loss_pred

# run_data = []
# for i in range(30):
#     print(1,i)
#     ppo  = PPO(logger, bc_loss = "MSE", parallel = 990, horizon = 20)
#     ppo.train_bc(e_states,e_actions, geometric = False, train_step = 1500, progress = True)
#     reward = evaluate(ppo.policy, env)
#     print(reward)
#     model_losses = test_model(states[np.random.permutation(states.shape[0])[:990]] , ppo)
#     run_data.append((reward, model_losses))

# run_data2 = []
# for i in range(30):
#     print(2, i)
#     ppo  = PPO(logger, bc_loss = "MSE", parallel = 990, horizon = 20)
#     ppo.train_bc(e_states,e_actions, geometric = False, train_step = 1500, progress = True)
#     reward = evaluate(ppo.policy, env)
#     print(reward)
#     model_losses = test_model(e_states , ppo)
#     run_data2.append((reward, model_losses))
    
# run_data3 = []
# for i in range(30):
#     print(3,i)
#     ppo  = PPO(logger, bc_loss = "logprob", parallel = 990, horizon = 20)
#     ppo.train_bc(e_states,e_actions, geometric = False, train_step = 1500, progress = True)
#     reward = evaluate(ppo.policy, env)
#     print(reward)
#     model_losses = test_model(e_states , ppo)
#     run_data3.append((reward, model_losses))
    
# run_data4 = []
# for i in range(30):
#     print(4,i)
#     ppo  = PPO(logger, bc_loss = "logprob", parallel = 990, horizon = 20)
#     ppo.train_bc(e_states,e_actions, geometric = True, train_step = 1500, progress = True)
#     reward = evaluate(ppo.policy, env)
#     print(reward)
#     model_losses = test_model(e_states , ppo)
#     run_data4.append((reward, model_losses))

run_data5 = []
for i in range(30):
    print(4,i)
    ppo  = PPO(logger, bc_loss = "logprob", parallel = 64, horizon = 20)
    ppo.train_bc(e_states,e_actions, geometric = True, train_step = 1500, progress = True)
    reward = evaluate(ppo.policy, env)
    print(reward)
    model_losses = test_model(np.asarray([env.reset() for _ in range(64)]) , ppo)
    run_data5.append((reward, model_losses))
    

run_data6 = []
for i in range(30):
    print(4,i)
    ppo  = PPO(logger, bc_loss = "MSE", parallel = 64, horizon = 20)
    ppo.train_bc(e_states,e_actions, geometric = True, train_step = 1500, progress = True)
    reward = evaluate(ppo.policy, env)
    print(reward)
    model_losses = test_model(np.asarray([env.reset() for _ in range(64)]) , ppo)
    run_data6.append((reward, model_losses))
    
        
    
print(np.asarray([np.asarray(run_data)[:,0][i][0] for i in range(30)]).mean(),
      np.asarray([np.asarray(run_data)[:,0][i][0] for i in range(30)]).std())
print(np.asarray([np.asarray(run_data2)[:,0][i][0] for i in range(30)]).mean(),
      np.asarray([np.asarray(run_data2)[:,0][i][0] for i in range(30)]).std())
print(np.asarray([np.asarray(run_data3)[:,0][i][0] for i in range(30)]).mean(),
      np.asarray([np.asarray(run_data3)[:,0][i][0] for i in range(30)]).std())
print(np.asarray([np.asarray(run_data4)[:,0][i][0] for i in range(30)]).mean(),
      np.asarray([np.asarray(run_data4)[:,0][i][0] for i in range(30)]).std())
print(np.asarray([np.asarray(run_data5)[:,0][i][0] for i in range(30)]).mean(),
      np.asarray([np.asarray(run_data5)[:,0][i][0] for i in range(30)]).std())
print(np.asarray([np.asarray(run_data6)[:,0][i][0] for i in range(30)]).mean(),
      np.asarray([np.asarray(run_data6)[:,0][i][0] for i in range(30)]).std())


print(np.asarray([np.asarray(np.asarray(run_data)[:,1][i]).mean() 
 for i in range(30)]).mean(), np.asarray([np.asarray(np.asarray(run_data)[:,1][i]).mean() 
 for i in range(30)]).std())
print(np.asarray([np.asarray(np.asarray(run_data2)[:,1][i]).mean() 
 for i in range(30)]).mean(), np.asarray([np.asarray(np.asarray(run_data2)[:,1][i]).mean() 
 for i in range(30)]).std())
print(np.asarray([np.asarray(np.asarray(run_data3)[:,1][i]).mean() 
 for i in range(30)]).mean(), np.asarray([np.asarray(np.asarray(run_data3)[:,1][i]).mean() 
 for i in range(30)]).std())
print(np.asarray([np.asarray(np.asarray(run_data4)[:,1][i]).mean() 
 for i in range(30)]).mean(), np.asarray([np.asarray(np.asarray(run_data4)[:,1][i]).mean() 
 for i in range(30)]).std())
print(np.asarray([np.asarray(np.asarray(run_data5)[:,1][i]).mean() 
 for i in range(30)]).mean(), np.asarray([np.asarray(np.asarray(run_data5)[:,1][i]).mean() 
 for i in range(30)]).std())
print(np.asarray([np.asarray(np.asarray(run_data6)[:,1][i]).mean() 
 for i in range(30)]).mean(), np.asarray([np.asarray(np.asarray(run_data6)[:,1][i]).mean() 
 for i in range(30)]).std())