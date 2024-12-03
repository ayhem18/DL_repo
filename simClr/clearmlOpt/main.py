
"""
This script contains the code for tuning the Pytorch Lightning module
"""
from clearml import Task, Logger
from clearml.automation.optimization import HyperParameterOptimizer

from clearml.automation import DiscreteParameterRange
from clearml.automation.optuna import OptimizerOptuna

from tune_task import opt_task


def func(arg1: int, 
         arg2:int):
    
    # initialize the tuning task
    opt_task(arg1, arg2)

    # get the task id
    template_task_id = Task.get_task(project_name='exp_hp_opt', 
                                    task_name=f'test_run').id # the template task should be already created at this point

    optimization_task = Task.init( 
                        project_name='HP_OPTIMIZATION',
                        task_name=f'test_run_hp_optimization',
                        task_type=Task.TaskTypes.optimizer,
                        reuse_last_task_id=False, 

                        auto_connect_arg_parser=False, 
                        auto_connect_frameworks=False, 
                        auto_connect_streams=False
                    )   
    
    # there should be some way to connect the optimization task to the optimizer
    # the one implicitly suggested by the tutorial: https://github.com/allegroai/clearml/blob/master/examples/optimization/hyper-parameter-optimization/hyper_parameter_optimizer.py
    # is by creating a dictionary object with the template_task_id, connecting it to the 'optimization_task' and using the same value (through the dictionary) in the HPOptimizer instance.

        
    args = {"template_task_id": template_task_id}

    # connect the 'optimization_task' with the 'args' variable
    optimization_task.connect(args)

    # define an optimizer
    tune_optimizer = HyperParameterOptimizer(
        base_task_id=args['template_task_id'], # calling the template_task_id through the 'args' seems like the only way to connect the hp optimizer to the 'optimization_task'

        # adding the 'General/' prefix, according to the tutorial: 
        # https://github.com/allegroai/clearml/blob/master/examples/optimization/hyper-parameter-optimization/hyper_parameter_optimizer.py

        hyper_parameters=[
            DiscreteParameterRange(name='General/x', values=list(range(1, 5))), 
        ],

        # this is the objective metric we want to maximize/minimize
        objective_metric_title='res',
        objective_metric_series='res',
        # minimize the val_epoch_loss
        objective_metric_sign='min',
        max_number_of_concurrent_tasks=1, #humble starts

        optimizer_class=OptimizerOptuna,
        save_top_k_tasks_only=3,      

        # optimization_time_limit=2, 
        # these are parameters passed to the Optuna Optimizer
        max_iteration_per_job=1,
        total_max_jobs=1
        )

    # run the optimizer
    tune_optimizer.start_locally()

    # to my understanding the only search algorithm that will halt automatically is the grid search after a complete search of the space 
    # the only way to stop non-grid algos is by setting a time limit
    tune_optimizer.set_time_limit(in_minutes=1)

    # make sure to call the 'wait' function as the experiments are running on background threads 
    # and not the main process...
    tune_optimizer.wait()

    # not setting the time limit for now
    tune_optimizer.stop()


if __name__ == '__main__':
    func(1, 2)
