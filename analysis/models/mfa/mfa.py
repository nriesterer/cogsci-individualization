""" MFA implementation stripped of its person-training capabilities.

"""

import random

import ccobra

class MFAModel(ccobra.CCobraModel):
    def __init__(self, name='MFAModel', k=1):
        super(MFAModel, self).__init__(name, ["syllogistic"], ["single-choice"])

        # Initialize member variables
        self.mfa_population = dict()

    def pre_train(self, dataset):
        # Iterate over subjects in the dataset
        for subj_data in dataset:
            # Iterate over the task for an individual subject
            for task_data in subj_data:
                # Create the syllogism object and extract the task and response encodings
                syllogism = ccobra.syllogistic.Syllogism(task_data['item'])
                encoded_task = syllogism.encoded_task
                encoded_response = syllogism.encode_response(task_data['response'])

                # Prepare the response counter for this task if not present already
                if encoded_task not in self.mfa_population:
                    self.mfa_population[encoded_task] = dict()

                # Increment the response count for the present task
                self.mfa_population[encoded_task][encoded_response] = \
                    self.mfa_population[encoded_task].get(encoded_response, 0) + 1

    def get_mfa_prediction(self, item, mfa_dictionary):
        # Extract the encoded task
        syllogism = ccobra.syllogistic.Syllogism(item)
        encoded_task = syllogism.encoded_task
        encoded_choices = [syllogism.encode_response(x) for x in item.choices]

        if encoded_task in mfa_dictionary:
            # Extract the potential MFA responses which are allowed in terms
            # of the possible response choices
            potential_responses = []
            for response, count in mfa_dictionary[encoded_task].items():
                if response in encoded_choices:
                    potential_responses.append((response, count))

            # If potential responses are available, determine the one with
            # maximum frequency
            if potential_responses:
                max_count = -1
                max_responses = []
                for response, count in potential_responses:
                    if count > max_count:
                        max_count = count
                        max_responses = []

                    if count >= max_count:
                        max_responses.append(response)

                # In case of ties, draw the MFA response at random from the options
                # with maximum frequency
                encoded_prediction = max_responses[random.randint(0, len(max_responses) - 1)]
                return encoded_prediction

        # If no MFA response is available, return None
        return None

    def predict(self, item, **kwargs):
        # Create the syllogism object
        syllogism = ccobra.syllogistic.Syllogism(item)

        # Return the population MFA if available
        population_prediction = self.get_mfa_prediction(item, self.mfa_population)
        if population_prediction is not None:
            return syllogism.decode_response(population_prediction)

        # Return a random response if no MFA data is available
        return item.choices[random.randint(0, len(item.choices) - 1)]
