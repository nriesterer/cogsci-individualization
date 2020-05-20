import numpy as np

import ccobra

class MFAModel(ccobra.CCobraModel):
    def __init__(self, name='MFA'):
        super(MFAModel, self).__init__(name, ['syllogistic'], ['single-choice'])

    def pre_train(self, database, **kwargs):
        responses = np.zeros((64, 9))
        for subj_data in database:
            for task_data in subj_data:
                syllogism = ccobra.syllogistic.Syllogism(task_data['item'])
                task_idx = ccobra.syllogistic.SYLLOGISMS.index(syllogism.encoded_task)
                resp_idx = ccobra.syllogistic.RESPONSES.index(
                    syllogism.encode_response(task_data['response']))

                responses[task_idx, resp_idx] += 1
        self.mfa = responses.argmax(axis=1)

    def predict(self, item, **kwargs):
        syllogism = ccobra.syllogistic.Syllogism(item)
        syl_idx = ccobra.syllogistic.SYLLOGISMS.index(syllogism.encoded_task)
        pred_idx = self.mfa[syl_idx]
        pred = ccobra.syllogistic.RESPONSES[pred_idx]
        return syllogism.decode_response(pred)
