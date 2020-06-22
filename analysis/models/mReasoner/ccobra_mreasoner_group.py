""" CCOBRA model wrapper for mReasoner.

"""

import sys
import collections
import threading

import ccobra
import mreasoner
import numpy as np

import time

import logging

logging.basicConfig(level=logging.INFO)

class CCobraMReasoner(ccobra.CCobraModel):
    """ mReasoner CCOBRA model implementation.

    """

    def __init__(self, name='mReasoner-Group', fit_its=0, n_samples=2, method='grid', num_threads=4):
        """ Initializes the CCOBRA model by launching the interactive LISP subprocess.

        """

        super(CCobraMReasoner, self).__init__(name, ['syllogistic'], ['single-choice'])

        # Launch mreasoner interface
        self.cloz = mreasoner.ClozureCL()
        self.mreas_path = mreasoner.source_path()

        self.mreasoner = mreasoner.MReasoner(self.cloz.exec_path(), self.mreas_path)

        # Member variables
        self.method = method
        self.num_threads = num_threads
        self.fit_its = fit_its
        self.n_samples = n_samples

        # Adaption parameters
        self.adapt_x = []
        self.adapt_y = []
        self.old_params = [self.mreasoner.default_params[value] for value in ['epsilon', 'lambda', 'omega', 'sigma']]

        self.cnt = 0
        self.adapt_cnt = 0

        self.start_time = None

    def __deepcopy__(self, memo):
        """ Custom deepcopy required because thread locks cannot be pickled. Deepcopy realized by
        creating a fresh instance of the mReasoner model and syncing parameters.

        Parameters
        ----------
        memo : dict
            Memo dictionary of objects already copied. Should be passed to nested deepcopy calls.

        Returns
        -------
        CCobraMReasoner
            Copied object instance.

        """

        self.cnt += 1
        #print('Copy', self.cnt)

        new = CCobraMReasoner(
            self.name, self.fit_its, n_samples=self.n_samples, method=self.method,
            num_threads=self.num_threads)

        # Deep copy properties of mreasoner instance
        for param, value in self.mreasoner.params.items():
            new.mreasoner.set_param(param, value)

        return new

    def end_participant(self, subj_id, **kwargs):
        """ When the prediction phase is finished, terminate the LISP subprocess.

        """

        # Parameterization output
        print('Fit ({:.2f}s, {} its) id={} params={}'.format(
            time.time() - self.start_time,
            self.fit_its,
            subj_id,
            str(list(self.mreasoner.params.items())).replace(' ', ''),
        ))
        sys.stdout.flush()

        self.mreasoner.terminate()

    def start_participant(self, **kwargs):
        self.id = kwargs['id']
        self.start_time = time.time()

    def pre_train(self, dataset):
        """ Pre-trains the model by fitting mReasoner.

        Parameters
        ----------
        dataset : list(list(dict(str, object)))
            Training data.

        """

        if self.fit_its == 0:
            return

        train_x = []
        train_y = []
        for subj_data in dataset:
            for task_data in subj_data:
                item = task_data['item']
                enc_task = ccobra.syllogistic.encode_task(item.task)
                enc_resp = ccobra.syllogistic.encode_response(task_data['response'], item.task)
                train_x.append(enc_task)
                train_y.append(enc_resp)

        # Perform the fitting
        print('Pretrain...')
        self.fit_mreasoner_grid_parallel(train_x, train_y, self.fit_its, self.num_threads)

    def fit_mreasoner_grid_parallel(self, train_x, train_y, fit_its, num_threads):
        print('grid fitting...')
        sys.stdout.flush()

        thread_results = []
        def work_fun(train_x, train_y, mreas_path, cloz_path, epsilon_values, fit_its, n_samples):
            # Create local mReasoner copy
            thread_mreasoner = mreasoner.MReasoner(cloz_path, mreas_path)

            best_score = 0
            best_params = []

            for p_epsilon in epsilon_values:
                for p_lambda  in np.linspace(*thread_mreasoner.param_bounds[1], fit_its):
                    for p_omega in np.linspace(*thread_mreasoner.param_bounds[2], fit_its):
                        for p_sigma in np.linspace(*thread_mreasoner.param_bounds[3], fit_its):
                            params = [p_epsilon, p_lambda, p_omega, p_sigma]
                            thread_mreasoner.set_param_vec(params)

                            preds = {}
                            for syllog in ccobra.syllogistic.SYLLOGISMS:
                                preds[syllog] = []

                                for _ in range(n_samples):
                                    preds[syllog].append(thread_mreasoner.query(syllog))

                            hits = 0
                            for task, truth in zip(train_x, train_y):
                                for pred in preds[task]:
                                    if pred == truth:
                                        hits += 1 / len(preds[task])

                            if hits > best_score:
                                best_score = hits
                                best_params = [params]
                            elif hits == best_score:
                                best_params.append(params)

            thread_results.append((best_score, best_params))
            thread_mreasoner.terminate()

        epsilon_values = [
            [0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8],
            [0.9, 1.0]
        ]

        threads = []
        for epsilon_conf in epsilon_values:
            th = threading.Thread(target=work_fun, args=(
                train_x, train_y, self.mreas_path, self.cloz.exec_path(), epsilon_conf, fit_its, self.n_samples))
            th.start()
            threads.append(th)

        for th in threads:
            th.join()

        # Evaluate the results
        best_score_value = np.max([score for score, _ in thread_results])
        best_params = []
        for res_score, res_params in thread_results:
            if res_score == best_score_value:
                best_params.extend(res_params)
        self.best_params = best_params

        used_params = best_params[np.random.randint(0, len(best_params))]
        self.mreasoner.set_param_vec(used_params)

        print('training done.')
        sys.stdout.flush()

    def pre_train_person(self, dataset, **kwargs):
        """ Person training is deactivated for the group model.

        """

        return

    def predict(self, item, **kwargs):
        """ Queries mReasoner for a prediction.

        Parameters
        ----------
        item : ccobra.Item
            Task item.

        Returns
        -------
        list(str)
            Syllogistic response prediction.

        """

        enc_task = ccobra.syllogistic.encode_task(item.task)

        enc_resp_cands = []
        for _ in range(self.n_samples):
            pred = self.mreasoner.query(enc_task)
            if isinstance(pred, list):
                enc_resp_cands.extend(pred)
            else:
                enc_resp_cands.append(pred)

        pred_counts = dict(collections.Counter(enc_resp_cands))
        max_count = np.max(list(pred_counts.values()))
        max_preds = [pred for pred, pred_count in pred_counts.items() if pred_count == max_count]

        enc_resp = np.random.choice(max_preds)

        return ccobra.syllogistic.decode_response(enc_resp, item.task)
