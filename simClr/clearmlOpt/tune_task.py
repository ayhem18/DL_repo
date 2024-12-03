from clearml import Task, Logger

def opt_task(arg1:int, 
            arg2: int):

    # create a task
    task = Task.init(project_name='exp_hp_opt',
                    task_name='test_run'
                    )
    
    logger:Logger = task.get_logger()

    # there is only one hyperparameter 
    args = {"x": None, 'arg1': arg1, 'arg2': arg2}

    task.connect(args)

    total = 0

    if args['x'] is None or args['x'] <= 0 :
        total = 10 ** 4

    else:
        for i in range(args['x']):
            v = (i - args['x']) ** 2
            total += v
            logger.report_scalar(title='inter_res', series='inter_res', value=v, iteration=i + 1)

        total /= args['x']
    

    logger.report_scalar(title='arg1', series='arg1', value=args['arg1'], iteration=1)
    logger.report_scalar(title='arg2', series='arg2', value=args['arg2'], iteration=1)
    logger.report_scalar(title='res', series='res', value=total, iteration=1)
    
    # close the task ??
    task.close()
