# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from experiment_master_file import experiments_meta_dict
from backend_definitions_dict import BACKEND_DEFS
from blackbox_helper import (
    get_transfer_points_active,
    do_tasks_in_order,
    get_configs,
)

# You will have to run the two generate_[...].py scripts in simopt/

experiment = "SimOpt"
optimiser_set = [("BoundingBox", "Transfer")]
experiment_meta_data = experiments_meta_dict[experiment]
backend = experiment_meta_data["backend"]
optimiser, optimiser_type = optimiser_set[0]

simopt_backend_file = experiment_meta_data["simopt_backend_file"]
yahpo_dataset, yahpo_scenario = None, None
xgboost_res_file = None

metric_def, opt_mode, active_task_str, uses_fidelity = BACKEND_DEFS[backend]
full_task_list, get_backend = get_configs(
    backend, xgboost_res_file, simopt_backend_file, yahpo_dataset, yahpo_scenario
)

# Some settings, set to small values
seed = 0
active_task_list = full_task_list[:2]
pte_func = get_transfer_points_active
metric = metric_def
points_per_task = 3


res = do_tasks_in_order(
    seed=seed,
    active_task_list=active_task_list,
    pte_func=pte_func,
    points_per_task=points_per_task,
    get_backend=get_backend,
    optimiser=optimiser,
    metric=metric,
    opt_mode=opt_mode,
    active_task_str=active_task_str,
    uses_fidelity=uses_fidelity,
    n_workers=4,
)
