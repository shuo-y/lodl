from PThenO import PThenO
import pandas as pd
import sqlite3
import torch
import numpy as np
from torch.distributions.categorical import Categorical
import pdb
from datetime import datetime
import os
import random
from functools import cmp_to_key

class VMScheduling(PThenO):

    def __init__(
        self,
        num_feature=10,
        num_history=100,
        num_host=10,
        rand_seed=0,
        num_train=200,
        num_eval=10,
        num_test=200,
        num_per_instance=100
    ):
        super(VMScheduling, self).__init__()
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)

        con = sqlite3.connect("data/packing_trace_zone_a_v1.sqlite")

        self.process_data(num_train, num_eval, num_test, num_per_instance)



    def draw_hist(self, data, filename):
        import matplotlib.pyplot as plt
        plt.hist(data)
        plt.savefig(filename)

    def gen_data_feat(
        self,
        data_feat,
        data_df,
        beg_time,
        num_type,
        day_divide,
        num_startinterval,
        num_durinterval,
        dur_time_unit,
    ):
        """
        data_feat should be a torch tensor with (num_type, num_startinterval* num_durinterval)
        Generate a feature from a data frame
        # types * # start_interval
        """

        # Don't make the data_feat too big'
        start_time_unit = (1 / day_divide) / num_startinterval

        for i in range(len(data_df)):
            vm_type_id = (int)(data_df.iloc[i]['vmTypeId'])
            start_time = data_df.iloc[i]['starttime'] - beg_time
            start_bucket = (int)(start_time // start_time_unit)
            if start_bucket >= num_startinterval:
                raise ValueError("start bucket is too big")

            #If raise error here please check if start_time >> start_time_unit * num_startinterval and if the vm_type_id is bigger than num_type

            dur_time = data_df.iloc[i]['endtime'] - data_df.iloc[i]['starttime']
            dur_bucket = (int)(dur_time // dur_time_unit)
            if dur_bucket >= num_durinterval:
                dur_bucket = num_durinterval - 1
            data_feat[vm_type_id][start_bucket * num_durinterval + dur_bucket] += 1

        return

    def process_data(
        self,
        num_train,
        num_eval,
        num_test,
        num_per_instance,
        day_divide=96,
        num_startinterval=30,
        num_durinterval=50,
        dur_time_unit=0.01,
    ):
        """
         1 / 1440    1 mins
         1 / 360     3 mins
         1 / 160     9 mins
         1 / 96      15 mins
         1 / 64      22.5 mins
         1 / 32      45 mins
         1 / 16 = 0.615  90 mins is an integer
        split data by hours
        """
        now1 = datetime.now()
        num_days = 14
        intervals = [(i/day_divide, (i + 1)/day_divide)  for i in range(day_divide)]
        for day in range(1, num_days):
            intervals.extend([(day + (i/day_divide), day + ((i + 1)/day_divide))  for i in range(day_divide)])

        con = sqlite3.connect("data/packing_trace_zone_a_v1.sqlite")
        self.vmtype_df = pd.read_sql_query("SELECT * from vmType", con)
        max_type_id = max(self.vmtype_df['vmTypeId'].values.tolist())
        min_type_id = min(self.vmtype_df['vmTypeId'].values.tolist())
        # Type id from 0 to max_type_id
        self.num_job_types = max_type_id - min_type_id + 1
        self.num_per_instance = num_per_instance

        self.core_cap = np.zeros(self.num_job_types)
        self.mem_cap = np.zeros(self.num_job_types)
        for type_id in range(self.num_job_types):
            self.core_cap[type_id] = self.vmtype_df.loc[self.vmtype_df['vmTypeId'] == type_id]['core'].iloc[0]
            self.mem_cap[type_id] = self.vmtype_df.loc[self.vmtype_df['vmTypeId'] == type_id]['memory'].iloc[0]


        print("query sql...")
        vm_req_frames_1 = pd.read_sql_query("SELECT * from vm WHERE starttime >= 0.0 AND starttime < 7.0", con)
        vm_req_frames_2 = pd.read_sql_query("SELECT * from vm WHERE starttime >= 7.0 AND starttime < 14.0", con)
        # There are 14-day trace based on https://github.com/Azure/AzurePublicDataset/blob/master/AzureTracesForPacking2020.md
        # Just takes twice and then concat and not to make it too big
        # Check list if non decreasing
        # https://stackoverflow.com/questions/4983258/check-list-monotonicity

        self.vm_req_frames = pd.concat([vm_req_frames_1, vm_req_frames_2], ignore_index=True)
        self.vm_req_frames = self.vm_req_frames.dropna()

        self.max_jobs_per_type = 0

        print("sort jobs by starttime, priority, endttime, ")
        self.vm_req_frames = self.vm_req_frames.sort_values(by=['starttime', 'priority', 'endtime'])
        # pandas sort list
        #torchfeat_file = f"data/vm_daydivide{day_divide}_{num_data_frames}_begint_{num_startinterval}_durint_{num_durinterval}_{dur_time_unit}torchfeat.pt"
        start_time_unit = (1 / day_divide) / num_startinterval
        print("The normalized (1 is one day) start_time_unit:", start_time_unit)

        if ((2 * (num_train + num_eval + num_test) * num_per_instance) >  self.vm_req_frames.shape[0]):
            raise ValueError("There are not enough trace data for num_train, num_eval, and num_test, reduce the num_train, num_eval, and num_test")


        total_ins = 2 * (num_train + num_eval + num_test)
        self.feat = torch.zeros(total_ins, self.num_job_types)

        vm_rows = self.vm_req_frames.shape[0]
        self.vm_offset = random.randint(0, vm_rows - total_ins * num_per_instance)

        for i in range(total_ins):
            type_list = self.vm_req_frames.iloc[(self.vm_offset + i * num_per_instance): (self.vm_offset + (i + 1) * num_per_instance)]['vmTypeId'].values.tolist()
            self.feat[i] = torch.histogram(torch.tensor(type_list).float(), bins=max_type_id - min_type_id + 1, range=(min_type_id, max_type_id + 1), density=True)[0]


        self.trainxidx = [2 * i for i in range(0, num_train)]
        self.trainyidx = [2 * i + 1 for i in range(0, num_train)]

        self.evalxidx = [2 * i for i in range(num_train, (num_train + num_eval))]
        self.evalyidx = [2 * i + 1 for i in range(num_train, (num_train + num_eval))]

        self.testxidx = [2 * i for i in range((num_train + num_eval), (num_train + num_eval + num_test))]
        self.testyidx = [2 * i + 1 for i in range((num_train + num_eval), (num_train + num_eval + num_test))]


        self.trainX = self.feat[self.trainxidx]
        self.trainY = self.feat[self.trainyidx]

        self.evalX = self.feat[self.evalxidx]
        self.evalY = self.feat[self.evalyidx]

        self.testX = self.feat[self.testxidx]
        self.testY = self.feat[self.testyidx]

        now2 = datetime.now()
        print("init data time", now2 - now1)


    def get_train_data(self, **kwargs):
        return self.trainX, self.trainY, self.trainyidx

    def get_val_data(self, **kwargs):
        return self.evalX, self.evalY, self.evalyidx

    def get_test_data(self, **kwargs):
        return self.testX, self.testY, self.testyidx


    def interval2ind(self, feat: np.ndarray):
        """
        Given the input virtual machine feature convert it to output indicator
        col 0 : start time
        col 1 : end time

        """

        indi = []
        timestamp = []
        for i in range(len(feat)):
            timestamp.append((feat[i][0], 2, [i])) # 2 represent start
            timestamp.append((feat[i][1], 1, [i])) # 1 represent end

        timestamp.sort()
        timestamp_compress = []
        timestamp_compress.append(timestamp[0])

        for i in range(1, len(timestamp)):
            prev = len(timestamp_compress) - 1
            if (timestamp[i][0] == timestamp_compress[prev][0]
                and timestamp[i][1] == timestamp_compress[prev][1]):
                timestamp_compress[prev][2].append(timestamp[i][2][0])
            else:
                timestamp_compress.append(timestamp[i])

        ind_mat = np.zeros((len(feat), len(timestamp_compress)))

        for i in range(len(timestamp_compress)):
            if timestamp_compress[i][1] == 2:
                # If start set to 1
                for idx in timestamp_compress[i][2]:
                    ind_mat[idx][i:] = 1
            if timestamp_compress[i][1] == 1:
                # if end set to 1
                for idx in timestamp_compress[i][2]:
                    ind_mat[idx][i:] = 0

        return ind_mat

    def hist2int(
        self,
        time_probs: torch.Tensor,
        arr_max: float,
        dur_max: float,
        num_jobs: int,

    ):
        """
        probs is the probs of 2 dimensions array num of arrival bins X num of duration bins
        output is a sampled feature with start time and end time
        """

        time_feat = np.zeros((num_jobs, 2))
        arr_sampler = Categorical(probs=time_probs.sum(dim = 1))
        dur_samplers = [Categorical(probs=time_probs[i]) for i in range(len(time_probs))]

        num_arr_bin = len(time_probs)
        num_dur_bin = len(time_probs[0])

        output = np.zeros((num_jobs, 2))

        for i in range(num_jobs):
            arr_bin = arr_sampler.sample().item()
            dur_bin = dur_samplers[arr_bin].sample().item()
            arr_time = arr_max * (arr_bin / num_arr_bin)
            dur_time = dur_max * (dur_bin / num_dur_bin)
            output[i][0] = arr_time
            output[i][1] = dur_time

        return output


    def pick_best_fit(self, job_cap, bins):
        inds = [(bins[i], i) for i in range(len(bins))]

        inds.sort()
        for i in range(len(inds) - 1, -1, -1):
            if inds[i][0] + job_cap <= 1:
                bins[inds[i][1]] += job_cap
                return inds[i][1]

        return -1

    def pick_best_fit_from2bins(self, job_cap_1, job_cap_2, bins1, bins2):
        # Pick bins for best fit but for two types
        inds = [(bins1[i], bins2[i], i) for i in range(len(bins1))]
        inds.sort()

        for i in range(len(inds)):
            if inds[i][0] + job_cap_1 <= 1 and inds[i][1] + job_cap_2 <= 1:
                bins1[inds[i][2]] += job_cap_1
                bins2[inds[i][2]] += job_cap_2
                return inds[i][2]
        return -1


    def get_best_fit(self, jobs, job_type, decision, num_machines):
        """
        Given jobs
        return best fit decisions of a single types
        col 0 is start time, col 1 is end time
        """
        inds = [(jobs[job_type][i][0], 1, i) for i in range(len(jobs[job_type])) if jobs[job_type][i] > 0] # 1 means start
        inds.extend([(jobs[job_type][i][1], 2, i) for i in range(len(jobs[job_type])) if jobs[job_type][i] > 0]) # 2 means stop

        bins1 = np.zeros(num_machines)
        bins2 = np.zeros(num_machines)
        cur_job_set = dict()

        des_ind = 0
        inds.sort()

        for i in range(len(inds)):
            job_ind = inds[i][2]
            if inds[i][1] == 1:
                # jobs[job_ind][2] means machine capacity
                machine_ind = self.pick_best_fit_from2bins(self.core_cap[job_type], self.mem_cap[job_type], bins1, bins2)
                ## return -1 means reject the job
                decision[des_ind] = machine_ind
                des_ind += 1
                if machine_ind != -1:
                    cur_job_set[job_ind] = machine_ind
            elif inds[i][1] == 2:
                # stop job
                if job_ind in cur_job_set:
                    pmachine_ind = cur_job_set[job_ind]
                    del cur_job_set[job_ind]
                    bins1[machine_ind] -= self.core_cap[job_type]
                    bins2[machine_ind] -= self.mem_cap[job_type]
                    # Calculate packing density
        return

    def get_obj_from_decision(self, jobs, job_type, decision, num_machines):
        """
        Given jobs and decisions
        return how well the decisions work
        """
        inds = [(jobs[job_type][i][0], 1, i) for i in range(len(jobs[job_type])) if jobs[job_type][i] > 0] # 1 means start
        inds.extend([(jobs[job_type][i][1], 2, i) for i in range(len(jobs[job_type])) if jobs[job_type][i] > 0]) # 2 means stop

        bins1 = np.zeros(num_machines)
        bins2 = np.zeros(num_machines)
        cur_job_set = dict()

        des_ind = 0
        inds.sort()

        obj = 0
        for i in range(len(inds)):
            job_ind = inds[i][2]
            if inds[i][1] == 1:
                # jobs[job_ind][2] means machine capacity
                if decision[des_ind] != -2:
                    machine_ind = decision[des_ind]
                    bins1[machine_ind] += self.core_cap[job_type]
                    bins2[machine_ind] += self.mem_cap[job_type]
                    des_ind += 1
                    if machine_ind != -1:
                        cur_job_set[job_ind] = machine_ind
                        obj -= 1
                else:
                    machine_ind = self.pick_best_fit_from2bins(self.core_cap[job_type], self.mem_cap[job_type], bins1, bins2)
                    # if the decision does not give us any ideas use the best fit greedy algorithm
                    if machine_ind != -1:
                        cur_job_set[job_ind] = machine_ind
                        obj -= 1
            elif inds[i][1] == 2:
                # stop job
                if job_ind in cur_job_set:
                    pmachine_ind = cur_job_set[job_ind]
                    del cur_job_set[job_ind]
                    bins1[machine_ind] -= self.core_cap[job_type]
                    bins2[machine_ind] -= self.mem_cap[job_type]
                    # Calculate packing density
        return obj

    def yi2decision(self, yi, num_machines):
        """
        Convert yi into [num_job_types, max_num_jobs, 2(beg time, end time)]
        """
        jobs = np.full((self.num_job_types, self.max_jobs_per_type, 2), fill_value=-1, dtype=int)
        decisions = np.full((self.num_job_types, self.max_jobs_per_type), fill_value=-2, dtype=int)

        for job_t in range(len(yi)):
            feat_ind = 0
            yfeat = yi[job_t].reshape(self.feat_shape[1], self.feat_shape[2])
            for beg_bucket in range(self.feat_shape[1]):
                for dur_bucket in range(self.feat_shape[2]):
                    job_num = int(yi[beg_bucket][dur_bucket] * self.max_jobs_per_type)
                    jobs[job_t][feat_ind][0] = beg_bucket
                    jobs[job_t][feat_ind][1] = beg_bucket + dur_bucket
                    feat_ind += 1
                    #If thow error here check if feat_ind[job_t] >= self.max_jobs_per_type
            self.get_best_fit(jobs[job_t], job_t, decisions[job_t], num_machines)

        return decisions

    def get_yi_obj(self, yi, zi, num_machines):
        jobs = np.full((self.num_job_types, self.max_jobs_per_type, 2), fill_value=-1, dtype=int)

        for job_t in range(len(yi)):
            feat_ind = 0
            yfeat = yi[job_t].reshape(self.feat_shape[1], self.feat_shape[2])
            for beg_bucket in range(self.feat_shape[1]):
                for dur_bucket in range(self.feat_shape[2]):
                    job_num = int(yi[beg_bucket][dur_bucket] * self.max_jobs_per_type)
                    jobs[job_t][feat_ind][0] = beg_bucket
                    jobs[job_t][feat_ind][1] = beg_bucket + dur_bucket
                    feat_ind += 1
                    #If thow error here check if feat_ind[job_t] >= self.max_jobs_per_type
            self.get_obj_from_decision(jobs[job_t], job_t, zi[job_t], num_machines)


    def get_single_bestfit(self, yi, trace_id):
        bins = []

        timestamp = 0
        # What if here invalid avoid invalid input
        # clamp is based on https://blog.csdn.net/qq_37388085/article/details/127251550
        def clamp_probs(probs):
            eps = torch.finfo(probs.dtype).eps
            return probs.clamp(min=eps, max=1 - eps)

        vm_dist = Categorical(clamp_probs(yi))
        cur_obj = 0

        # Suppose we only predict the work type , the arrival time and dur is known from the trace_id
        # decisions = -1 means allocate one new machine
        # decisions = Z+ menas allocate to any physical machine

        decisions = []

        sub_df = self.vm_req_frames.iloc[(self.vm_offset + trace_id * self.num_per_instance): (self.vm_offset + (trace_id + 1) * self.num_per_instance)]

        beg_time = sub_df['starttime'].iloc[0]
        cur_active_job = []
        # bin_timestamp = []
        # when allocate a new bin record the time
        # when a new bin become idea record the time
        # in this way calculate the numerator and denumerator
        # start, end, core_cap, mem_cap, bins_id

        def compare(x, y):
            if x[4] != y[4]:
                return x[4] < y[4]
            else:
                return x[1] < y[1]
        # Sort based on comparator
        # https://stackoverflow.com/questions/12749398/using-a-comparator-function-to-sort

        for i in range(self.num_per_instance):
            arr = sub_df['starttime'].iloc[i] - beg_time
            ter = sub_df['endtime'].iloc[i] - beg_time
            # now time

            new_cur_active_job = []
            sorted(cur_active_job, key=cmp_to_key(compare))
            for cnt, (start_time, end_time, core_cap, mem_cap, bins_id) in enumerate(cur_active_job):
                if end_time < arr:
                    bins[bins_id][0] -= core_cap
                    bins[bins_id][1] -= mem_cap
                if not (bins[bins_id][0] == 0 and bins[bins_id][1] == 0):
                    new_cur_active_job.append([start_time, end_time, core_cap, mem_cap, bins_id])

            cur_active_job = new_cur_active_job

            pred_vmt = vm_dist.sample()

            pred_mem_req = self.mem_cap[pred_vmt]
            pred_core_req = self.core_cap[pred_vmt]

            remain_cap = 1.0
            select_act = -1
            for p_act in range(len(bins)):
                # Consider the possible allocate machine
                if bins[p_act][0] + pred_core_req < 1.0 and bins[p_act][1] + pred_mem_req < 1.0:
                    pos_remain_cap = 1.0 - bins[p_act][0] - pred_core_req
                    if pos_remain_cap < remain_cap:
                        select_act = p_act
                        remain_cap = pos_remain_cap
                else:
                    continue
            if select_act == -1:
                bins.append([pred_core_req, pred_mem_req])
                cur_active_job.append([arr, ter, pred_core_req, pred_mem_req, len(bins) - 1])
            else:
                bins[p_act][0] += pred_core_req
                bins[p_act][1] += pred_mem_req
                cur_active_job.append([arr, ter, pred_core_req, pred_mem_req, p_act])
            decisions.append(select_act)

        return torch.tensor(decisions)

    def check_single_bestfit(self, yi, decision, trace_id, dogreedy=False):
        """
        Given a yi and decision should return the objective
        if do greedy is true the decision will be ignored and do pure best fit algorithm
        """

        sub_df = self.vm_req_frames.iloc[(self.vm_offset + trace_id * self.num_per_instance): (self.vm_offset + (trace_id + 1) * self.num_per_instance)]
        bins = []

        beg_time = sub_df['starttime'].iloc[0]
        cur_active_job = []
        vm_cum_time = 0
        ph_cum_time = 0

        def compare(x, y):
            if x[4] != y[4]:
                return x[4] < y[4]
            else:
                return x[1] < y[1]
        # Sort based on comparator
        # https://stackoverflow.com/questions/12749398/using-a-comparator-function-to-sort


        for i in range(self.num_per_instance):
            arr = sub_df['starttime'].iloc[i] - beg_time
            ter = sub_df['endtime'].iloc[i] - beg_time

            new_cur_active_job = []
            sorted(cur_active_job, key=cmp_to_key(compare))
            for cnt, (start_time, end_time, core_cap, mem_cap, bins_id) in enumerate(cur_active_job):
                if end_time < arr:
                    bins[bins_id][0] -= core_cap
                    bins[bins_id][1] -= mem_cap
                    vm_cum_time = vm_cum_time + (end_time - start_time) * core_cap


                if bins[bins_id][0] <= 0 and bins[bins_id][1] <= 0:
                    pass
                    #ph_cum_time = ph_cum_time + end_time - bins_beg_timestamp[bins_id]
                    #del bins_beg_timestamp[bins_id]
                    # How to make sure it deletes the latest one?
                else:
                    new_cur_active_job.append([start_time, end_time, core_cap, mem_cap, bins_id])
            cur_active_job = new_cur_active_job

            gold_vmt = sub_df['vmTypeId'].iloc[i]

            mem_req = self.mem_cap[gold_vmt]
            core_req = self.core_cap[gold_vmt]


            if dogreedy:
                remain_cap = 1.0
                select_act = -1
                for p_act in range(len(bins)):
                    # Consider the possible allocate machine
                    if bins[p_act][0] + core_req < 1.0 and bins[p_act][1] + mem_req < 1.0:
                        pos_remain_cap = 1.0 - bins[p_act][0] - core_req
                        if pos_remain_cap < remain_cap:
                            select_act = p_act
                            remain_cap = pos_remain_cap
                    else:
                        continue
                if select_act == -1:
                    bins.append([core_req, mem_req])
                    cur_active_job.append([arr, ter, core_req, mem_req, len(bins) - 1])
                    #bins_beg_timestamp[len(bins) - 1] = arr
                else:
                    bins[p_act][0] += core_req
                    bins[p_act][1] += mem_req
                    cur_active_job.append([arr, ter, core_req, mem_req, p_act])
                    #if p_act not in bins_beg_timestamp:
                    #    bins_beg_timestamp[p_act] = arr
            else:
                p_act = decision[i]
                if p_act >= 0 and p_act < len(bins) and bins[p_act][0] + core_req < 1.0 and bins[p_act][1] + mem_req < 1.0:
                    bins[p_act][0] += core_req
                    bins[p_act][1] += mem_req
                    cur_active_job.append([arr, ter, core_req, mem_req, p_act])
                    #if p_act not in bins_beg_timestamp:
                    #    bins_beg_timestamp[p_act] = arr
                else:
                    bins.append([core_req, mem_req])
                    cur_active_job.append([arr, ter, core_req, mem_req, len(bins) - 1])
                    #bins_beg_timestamp[len(bins) - 1] = arr

        sorted(cur_active_job, key=cmp_to_key(compare))
        for cnt, (start_time, end_time, core_cap, mem_cap, bins_id) in enumerate(cur_active_job):
            bins[bins_id][0] -= core_cap
            bins[bins_id][1] -= mem_cap
            vm_cum_time = vm_cum_time + (end_time - start_time) * core_cap
            #if bins[bins_id][0] == 0 and bins[bins_id][1] == 0:
            #    ph_cum_time = ph_cum_time + end_time - bins_beg_timestamp[bins_id]
            #    del bins_beg_timestamp[bins_id]
            # how to properly compute packing density?
            # May have issuesin this for loop

        #print("total used bins", len(bins))
        #print("vm_cum_time", vm_cum_time)
        #print("ph_cum_time", ph_cum_time)
        #print(decision[:5])

        # TODO temporarily using the total number of allocated physical machines as decision quality
        return -float(len(bins))


    def get_decision(self, Y, aux_data, is_train=True, **kwargs):
        # Given vm jobs return decisions
        # Y [# instaces, # histograms ]
        # features = [# start interval, # dur interval]
        # output Z [# instances, # |jobs| ]

        if Y.ndim == 2 and isinstance(aux_data, list):
            decisions_list = []
            for (yi, trace_id) in zip(Y, aux_data):
                decisions = self.get_single_bestfit(yi, trace_id)
                decisions_list.append(decisions)
            return torch.stack(decisions_list)
        elif Y.ndim == 2 and isinstance(aux_data, int):
            decisions_list = []
            for yi in Y:
                decisions = self.get_single_bestfit(yi, aux_data)
                decisions_list.append(decisions)
            return torch.stack(decisions_list)
        elif Y.ndim == 1:
            decisions = self.get_single_bestfit(Y, aux_data)
            return decisions
        else:
            print("Y.shape", Y.shape)
            raise ValueError("The dimensions of Y or aux_data is not supported")

    def get_objective(self, Y, Z, aux_data, dogreedy=False, **kwargs):
        """
        the aux_data is needed here for the trace inds
        """
        if Y.ndim == 2 and (dogreedy==True or (hasattr(Z, "ndim") and Z.ndim == 2 and (len(Y) == len(Z)))):
            if isinstance(aux_data, list):
                objs = []
                for cnt in range(len(Y)):
                    objs.append(self.check_single_bestfit(Y[cnt], Z[cnt], aux_data[cnt], dogreedy=dogreedy))
                return torch.tensor(objs)
            elif isinstance(aux_data, int):
                objs = []
                for cnt in range(len(Y)):
                    objs.append(self.check_single_bestfit(Y[cnt], Z[cnt], aux_data, dogreedy=dogreedy))
                return torch.tensor(objs)
        elif Y.ndim == 1:
            assert Z.ndim == 1
            obj = self.check_single_bestfit(Y, Z, aux_data, dogreedy=dogreedy)
            return torch.tensor(obj)
        else:
            print("Y.shape", Y.shape)
            print("Z.shape", Z.shape)
            raise ValueError("The dimensions of Y and Z is not supported")

    def get_output_activation(self):
        pass

    def get_modelio_shape(self):
        return self.Xs.shape[-1], 1

    def get_twostageloss(self):
        return 'mse'



if __name__ == '__main__':
    vm_prob = VMScheduling(rand_seed=0)


    # TODO packig density
    # some cases for testing

    trainX, trainY, Yaux = vm_prob.get_train_data()

    now1 = datetime.now()
    Zs = vm_prob.get_decision(trainY, aux_data=Yaux)
    now2 = datetime.now()
    print("get decision time", now2 - now1)
    obj = vm_prob.get_objective(trainY, Zs, Yaux)
    now3 = datetime.now()
    print("get objective time", now3 - now2)
    pdb.set_trace()







