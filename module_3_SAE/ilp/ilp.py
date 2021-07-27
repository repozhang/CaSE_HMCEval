"""ILP"""
from mip import Model, xsum, maximize, BINARY
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from itertools import product
from tqdm import tqdm
import time
class ILP:
    def __init__(self,ilp_data,vlambda,q):
        self.vlambda=vlambda
        self.q=q
        self.ilp_data=ilp_data

    def decision(self):
        input_file = self.ilp_data
        # for tab delimited use:
        df = pd.read_csv(input_file, delimiter = "\t")
        # print(df.columns)
        t = df['turn']
        a = df['machine_confidence']
        b = df['human_confidence']
        k = df['machine_effort']
        l = df['human_effort_predicted_avg']
        l_truth=df['human_effort_groundtruth']
        max_confidence_predicted = sum(b)
        time_cost_human_predicted = sum(l)
        time_cost_human = sum(l_truth)
        M=len(df)
        y = Model()
        x = [y.add_var(var_type=BINARY) for i in range(M)]
        # print(x)
        N=int(self.q*len(df))

        y.objective = maximize(
                                xsum(
                                    (a[i] * x[i]
                                    + b[i] * (1-x[i])
                                    - self.vlambda* k[i] * x[i]
                                    - self.vlambda* l[i] * (1-x[i]))
                                                for i in range(M)
                                )
                            )


        # condition
        y += xsum(x[i] for i in range(M)) >= M-N


        y.optimize()

        decision_list=[]
        if y.num_solutions:
            # stdout.write('\n')
            for i, v in enumerate(y.vars):
                # print(i,v,v.x)
                decision_list.append(int(v.x))


        assigned_to_human=round(M-np.sum(decision_list),0)
        ratio_human_opt=round((M-np.sum(decision_list))/M,4)

        out_confidence=float(xsum((a[i] * decision_list[i]
                                    +b[i] * (1-decision_list[i]))
                                                for i in range(M)))

        out_effort=float(xsum((k[i] * decision_list[i]
                                    +l[i] * (1-decision_list[i]))
                                                for i in range(M)))

        out_effort_truth=float(xsum((k[i] * decision_list[i]
                                    +l_truth[i] * (1-decision_list[i]))
                                                for i in range(M)))
        machine_predict = df['machine_predict']
        target = df['target']
        human_predict = df['target']

        # hmc output
        hmc_predict=[]
        for v1,v2,v3 in zip(decision_list,machine_predict,human_predict):
            # print('i,j,k',i,type(i),j,k)
            if int(v1) == 1:
                # print(v2)
                hmc_predict.append(v2)
            else:
                hmc_predict.append(v3)


        # per turn output
        analysis_turn=False
        if analysis_turn==True:
            target_1=[]
            hmc_predict_1=[]
            decision_1=[]
            mconf=[]
            hconf=[]
            meffort=[]
            heffort=[]

            for t1,t2,t3,t4,t5,t6,t7,t8 in zip(t,target,hmc_predict,decision_list,a,b,k,l_truth):
                if t1==1:
                    target_1.append(t2)
                    hmc_predict_1.append(t3)
                    decision_1.append(t4)
                    mconf.append(t5)
                    hconf.append(t6)
                    meffort.append(t7)
                    heffort.append(t8)
            out_confidence_1=float(xsum((mconf[i] * decision_1[i]
                                        +hconf[i] * (1-decision_1[i]))
                                                    for i in range(len(decision_1))))
            out_effort_1 = float(xsum((meffort[i] * decision_1[i]
                                           + heffort[i] * (1 - decision_1[i]))
                                          for i in range(len(decision_1))))
            out_effort_total=sum(heffort)
            print(out_confidence_1/len(decision_1),sum(mconf)/len(decision_1))
            print('all,human ratio,time cost ratio, time cost, total time cost')
            print(len(target_1),(len(target_1)-sum(decision_1))/len(target_1),out_effort_1/out_effort_total,out_effort_1,out_effort_total) # all, human ratio, time cost
            outdict1=classification_report(target_1,hmc_predict_1,digits=4)
            print(outdict1)


        # hmi_predict p,r,f1,acc
        outdict=classification_report(target, hmc_predict,digits=4,output_dict=True)
        # pearsonout=np.corrcoef(target, hmc_predict) # pearson
        # pearsonout1=np.corrcoef(target,machine_predict)
        # print(pearsonout,pearsonout1)

        p_0=outdict['0']['precision']
        r_0=outdict['0']['recall']
        f1_0=outdict['0']['f1-score']

        p_1=outdict['1']['precision']
        r_1=outdict['1']['recall']
        f1_1=outdict['1']['f1-score']

        p_m=outdict['macro avg']['precision']
        r_m=outdict['macro avg']['recall']
        f1_m=outdict['macro avg']['f1-score']

        p_w=outdict['weighted avg']['precision']
        r_w=outdict['weighted avg']['recall']
        f1_w=outdict['weighted avg']['f1-score']

        accuracy=outdict['accuracy']

        return out_confidence / max_confidence_predicted, out_effort / time_cost_human_predicted, p_m, p_w, r_m, r_w, f1_m, f1_w, accuracy, ratio_human_opt, assigned_to_human / M, p_1, p_0, r_1, r_0, f1_1, f1_0, out_effort_truth / time_cost_human


def calculate():
    start = time.time()
    q=[0.5]
    # q=[i for i in np.arange(0,1.05,0.05)]
    # vlambda=[i for i in np.arange(0,45.1,0.1)]
    vlambda=[4.6]
    # vlambda=[500]

    # mydata='/.../ilp/data/ilp_data_tcp_f2.tsv'
    # mydata='/.../ilp/data/ilp_data_mcp_f2.tsv'
    mydata='/.../ilp/data/ilp_data_trustscore_f2.tsv'
    # mydata='/.../ilp/data/ilp_data_trustscore_f1.tsv'
    outputfile = 'output/ilp_out.tsv'
    mylist=[]
    f=open(outputfile,'a')
    column_name=['predicted_confidence','predicted_time_cost','precision_macro','precision_micro',
                 'recall_macro','recall_micro','f1_macro','f1_micro',
                 'accuracy',
                 'N/M','lambda',
                 'human_ratio',
                 'precision_1','precision_0','recall_1','recall_0','f1_1','f1_0','real_time_cost']
    f.write('\t'.join(column_name)+'\n')
    for c,d in tqdm(list(product(vlambda,q))):
        ilp = ILP(mydata,c, d)
        out_confidence,out_effort,p_m,p_w,r_m,r_w,f1_m,f1_w,accuracy,ratio_human_opt,human_ratio,p_1,p_0,r_1,r_0,f1_1,f1_0,out_effort_truth = ilp.decision()
        f.write('\t'.join([str(i) for i in [out_confidence,out_effort,p_m,p_w,r_m,r_w,f1_m,f1_w,accuracy,ratio_human_opt,c,human_ratio,p_1,p_0,r_1,r_0,f1_1,f1_0,out_effort_truth]])+'\n')
        mylist.append(tuple([d,out_confidence,out_effort,p_m,p_w,r_m,r_w,f1_m,f1_w,accuracy,ratio_human_opt,c,human_ratio,p_1,p_0,r_1,r_0,f1_1,f1_0,out_effort_truth]))
    end = time.time()
    print('totaltime',end-start)
    return mylist


if __name__=='__main__':
    mylist=calculate()
    for i in mylist:
        print(*list(i))

