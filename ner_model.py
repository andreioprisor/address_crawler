import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
import torch.nn.functional as F
import deepl

label_list = ['O', 'StreetNumber', 'StreetName', 'Unit', 'Municipality', 'Province', 'PostalCode', 'Orientation', 'GeneralDelivery']
label_to_id = {k: i for i, k in enumerate(label_list)}
# Function for getting labeled results by infering the NER model on the input text
			
def get_labeled_results(address_tokens, translator=None):
	tokenizer = AutoTokenizer.from_pretrained('NER_checkpoint')
	model = AutoModelForTokenClassification.from_pretrained('NER_checkpoint', num_labels=len(label_list))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	if translator and len(address_tokens) < 20:
		translator = deepl.Translator('45739d32-4c11-45af-9d1e-06859d3dc182:fx')
		add = ' '.join(address_tokens)
		address_tokens = str(translator.translate_text(add,target_lang='EN-US')).split()
	# 	print('Used Deepl API for translation')

	# Tokenize the input text
	tokens = tokenizer(address_tokens, is_split_into_words=True, truncation=True, return_tensors='pt')
	tokens = {k: v.to(device) for k, v in tokens.items()}
	# Perform prediction
	with torch.no_grad():  # Disable gradient calculation for inference
		predictions = model(**tokens).logits
	# Get the highest probability label index for each token
	probs = F.softmax(predictions, dim=-1)

	predicted_label_indices = torch.argmax(probs, dim=2)
	probs_for_predicted_label = probs[0, torch.arange(predicted_label_indices.shape[1]), predicted_label_indices[0]].tolist()
	# Convert label indices to label names
	predicted_labels = [label_list[idx] for idx in predicted_label_indices[0].tolist()]

	# Decode the tokens to words, handling subword tokens
	words = [tokenizer.decode(token_id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for token_id in tokens['input_ids'][0]]
	predicted_label_indices[0].tolist()
	# Post-process `words` and `predicted_labels` to handle subword tokens
	processed_words = []
	processed_labels = []
	probs = []
	for word, label, prob in zip(words, predicted_labels, probs_for_predicted_label):
		if word == '' or word == ' ' or word == ',':  # Skip empty words and commas
			continue
		if word.startswith("##"):
			processed_words[-1] += word[2:]  # Append subword to the previous word
		else:
			probs.append(prob)
			processed_words.append(word)
			processed_labels.append(label)
# Create a DataFrame
	res = pd.DataFrame({'ner': processed_labels, 'words': processed_words, 'probability': probs})
	return res

# Function for grouping entities from the labeled results in the DataFrame
def group_entities(df):
	grouped_entities = []
	current_entity = None
	prev_label = 'start'
	for _, row in df.iterrows():
		word, label, probability = row['words'], row['ner'], row['probability']
		# Handle non-entity words
		if label == 'O':
			if current_entity is not None:
				current_entity['words'] = ' '.join(current_entity['words'])
				current_entity['probability'] = sum(current_entity['probabilities']) / len(current_entity['probabilities'])
				grouped_entities.append(current_entity)
				current_entity = None
			prev_label = label
			continue

		if label != prev_label:
			if current_entity is not None:
				current_entity['words'] = ' '.join(current_entity['words'])
				current_entity['probability'] = sum(current_entity['probabilities']) / len(current_entity['probabilities'])
				grouped_entities.append(current_entity)
			# Start a new entity
			current_entity = {'label': label, 'words': [word], 'probabilities': [probability]}
			prev_label = label
		elif current_entity is not None:
			current_entity['words'].append(word)
			current_entity['probabilities'].append(probability)
			prev_label = label
	# Add the last entity
	if current_entity is not None:
		current_entity['words'] = ' '.join(current_entity['words'])
		current_entity['probability'] = sum(current_entity['probabilities']) / len(current_entity['probabilities'])
		grouped_entities.append(current_entity)

	return grouped_entities


# Function for selecting the highest probability entities from the grouped entities
def select_highest_probability_entities(grouped_entities):
	highest_prob_entities = {}
	for entity in grouped_entities:
		label = entity['label']
		probability = entity['probability']
		words = entity['words']

		if label not in highest_prob_entities.keys() or probability > highest_prob_entities[label][1]:
			highest_prob_entities[label] = (words, probability)

	return highest_prob_entities


# Function for formatting the address by applying the previous functions
def format_address(address_text, translator=None):
	res = get_labeled_results(address_text, translator=translator)
	grouped_entities = group_entities(res)  # Make sure you've run the previous grouping function
	highest_prob_entities = select_highest_probability_entities(grouped_entities)
	formatted_add = {'COUNTRY': '-', 'PROVINCE': '-', 'CITY': '-', 'COUNTY': '-', 'ZIPCODE': '-', 'ROAD': '-', 'ROADNUMBER': '-'}
	for label, (words, probability) in highest_prob_entities.items():
		if label == 'StreetNumber' and words.isdigit():
			formatted_add['ROADNUMBER'] = words
		elif label == 'StreetName' and sum(c.isdigit() for c in words) < sum(c.isalpha() for c in words):
			formatted_add['ROAD'] = words
		elif label == 'Municipality':
			formatted_add['CITY'] = words
		elif label == 'Province':
			formatted_add['PROVINCE'] = words
		elif label == 'PostalCode':
			formatted_add['ZIPCODE'] = words
		elif label == 'Unit':
			formatted_add['UNIT'] = words
	return formatted_add

        