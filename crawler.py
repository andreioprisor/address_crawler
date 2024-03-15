from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup as bs
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from selenium.webdriver.support import expected_conditions as EC
import re
import time
from dotenv import load_dotenv
import deepl
from urllib.parse import urljoin
from ner_model import format_address
import requests
from lxml import etree
import json

translations_dict = {
    "de": ["impressum"],
    "cz": ["o", "kontakt", "otisk"],
    "es": ["acerca de", "contacto", "imprenta"],
    "fr": ["à propos", "contact", "empreinte"]
}

countries = ['de', 'cz', 'es', 'fr', 'uk', 'gb', 'us']

states = [
    'alabama', 'al', 'alaska', 'ak', 'arizona', 'az', 'arkansas', 'ar',
    'california', 'ca', 'colorado', 'co', 'connecticut', 'ct', 'delaware', 'de',
    'florida', 'fl', 'georgia', 'ga', 'hawaii', 'hi', 'idaho', 'id', 'illinois', 'il',
    'indiana', 'in', 'iowa', 'ia', 'kansas', 'ks', 'kentucky', 'ky', 'louisiana', 'la',
    'maine', 'me', 'maryland', 'md', 'massachusetts', 'ma', 'michigan', 'mi',
    'minnesota', 'mn', 'mississippi', 'ms', 'missouri', 'mo', 'montana', 'mt',
    'nebraska', 'ne', 'nevada', 'nv', 'new hampshire', 'nh', 'new jersey', 'nj',
    'new mexico', 'nm', 'new york', 'ny', 'north carolina', 'nc', 'north dakota', 'nd',
    'ohio', 'oh', 'oklahoma', 'ok', 'oregon', 'or', 'pennsylvania', 'pa',
    'rhode island', 'ri', 'south carolina', 'sc', 'south dakota', 'sd', 'tennessee', 'tn',
    'texas', 'tx', 'utah', 'ut', 'vermont', 'vt', 'virginia', 'va', 'washington', 'wa',
    'west virginia', 'wv', 'wisconsin', 'wi', 'wyoming', 'wy'
]

class Crawler:
	def __init__(self, urls_list):
		self.options = Options()
		self.options.add_argument("--headless")
		self.service = Service(ChromeDriverManager().install())
		self.options.add_argument("--lang=en")
		self.driver = webdriver.Chrome(service=self.service, options=self.options)
		self.driver.set_window_size(1270, 920)
		self.interest_links = ['about', 'contact', 'imprint', 'impressum']
		self.urls_list = urls_list
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
		self.model = AutoModelForSequenceClassification.from_pretrained('./Classification_checkpoint').to(self.device)
		self.current_url = None
		self.page_source = None	
	
	# Function to get the probabilities from the classification model
	def get_prediction(self, tokenizer, model, text):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		encoding = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=16)
		encoding = {k: v.to(device) for k, v in encoding.items()}
		outputs = model(**encoding)
		logits = outputs.logits
		probs = torch.nn.functional.softmax(logits, dim=-1)
		return probs.squeeze().tolist()
	
	# Function to find all texts in a page containing a UK post code or a general post code pattert
	# This approach works only for addreses that contain this type of post codes, but it can be extended to other countries
	# easily by adding the corresponding post code pattern and handling it in the if else block
	def test_patterns(self, elements):
		matches = []
		general_postcode_pattern = r"(?:\s|,|^|-)\d{5,7}(?=\s|,|$|-)"
		uk_pattern = r"[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}"
		for e in elements:
			text = e.get_text(separator=', ', strip = True)
			match = re.findall(uk_pattern, text)
			if len(match) == 1:
				matches.append((match, e))
			else:
				match = re.findall(general_postcode_pattern, text)
				if len(match) == 1:
					matches.append((match, e))
		return matches
	
	# Function to get all relevant text elements in a page from a list of html text tags
	# we also use xpath to get all the text elements that contain text and don't have children
	def get_relevant_text(self, translation=None):
		soup = bs(self.page_source, 'html.parser')
		text_elements = []
		text_elements.extend([e for e in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'span', 'font'])])
		exp = '//div[text()]'
		exp2 = '//*[text() and not(node())]'
		encoded = self.page_source
		tree = etree.HTML(encoded.encode('utf-8'))
		if tree:
			divs = [bs(etree.tostring(e, pretty_print=True, method="html").decode("utf-8"), 'html.parser') for e in tree.xpath(exp)]
			other = [bs(etree.tostring(e, pretty_print=True, method="html").decode("utf-8"), 'html.parser') for e in tree.xpath(exp2)]
			text_elements.extend(divs)
			text_elements.extend(other)
		filtered_elements = [t for t in text_elements if len(t.text) > 10 and len(t.text) < 350 and any(char.isdigit() for char in t.text)]
		filtered_elements = list(set(filtered_elements))
		return filtered_elements

	# function to parse the page and return  a list with the top 3 addresses by probability
	def parse_page2(self, translation=None):
		relevant_text = self.get_relevant_text()
		matches = self.test_patterns(relevant_text)
		addreses = []

		for match in matches:
			text_element = match[1]
			postcode = match[0][0]
			node_text = text_element.get_text(separator=', ', strip = True).replace(',,', ',')
			addreses.append((node_text, self.get_prediction(self.tokenizer, self.model, node_text)[1],  postcode))
			# This try except block is splitting the text elements with the split function from the bs4 library, then it finds the postcode part and then
			# concatenates all splits with the postcode part.Each text obtained is infered through the classification model and the result is appended to the
			# addreses list along with the postcode part and the probability
			try:
				split = text_element.get_text(separator=' \t\t\t\t\t\t ', strip = True).split(' \t\t\t\t\t\t ')
				postcode_part = next(s for s in split if postcode.strip() in s.strip())
				if postcode_part:
					addreses.append((postcode_part, self.get_prediction(self.tokenizer, self.model, postcode_part)[1], postcode))
					split.remove(postcode_part)
				for s in split:
					if len(s) < 90 and len(s) > 7  and any(char.isdigit() for char in s):
						potential_address = s + ', ' + postcode_part
						if len(potential_address) < 90 and len(potential_address) > 7:
							prediction = self.get_prediction(self.tokenizer, self.model, potential_address)
							addreses.append((potential_address, prediction[1], postcode))
			except Exception as e:
				pass

			elements = text_element.find_next_siblings()

			# here we do the same as above but we concatenate siblings with the element containing the postcode
			for element in elements:
				sibling = element.get_text(separator=', ', strip = True).replace(',,', ',')
				if len(sibling) < 90 and len(sibling) > 7 and any(char.isdigit() for char in sibling):
					potential_address = sibling + ', ' + text_element.get_text(separator=', ', strip = True).replace(',,', ',')
					if len(potential_address) < 90 and len(potential_address) > 7:
						prediction = self.get_prediction(self.tokenizer, self.model, potential_address)
						addreses.append((potential_address, prediction[1], postcode))

		addreses = list(set(addreses))
		addreses = sorted(addreses, key=lambda x: x[1], reverse=True)
		return addreses[0:3]

	# Function to parse all the relevant urls that can contain an address(eg. about, contact, imprint, impressum) with the parse_page2 function
	# and return the top 3 addresses by probability
	def parse_interesting_urls(self, translation=None):
		urls = []
		found_addresses = []
		soup = bs(self.page_source, 'html.parser')
		anchors = soup.find_all('a')
		# soup = bs(self.driver.page_source, 'html.parser')
		# anchors = soup.find_all('a')
		interest_links = self.interest_links
		if translation and translation in translations_dict.keys():
			interest_links = translations_dict[translation]
		for link in anchors:
			for interest_link in interest_links:
				if interest_link in link.text.lower():
					a = None
					try:
						a = link['href']
					except Exception as e:
						pass
					if a and 'mail' not in a:
						current = self.current_url
						full_url = urljoin(current, a)
						urls.append(full_url)
		if not urls:
			return None
		urls = list(set(urls))
		for link in urls:
			try:
				response = requests.get(link, timeout=20)
				self.current_url = response.url
				if response.ok == False:
					print(f'Loading {link}')
					self.driver.get(link)
					self.page_source = self.driver.page_source
				else:
					self.page_source = response.text
				print(f'Loaded {link}')
			except Exception as e:
				print(f'Error loading {self.current_url}')
				continue
			addresses = self.parse_page2()
			if addresses:
				found_addresses.extend(addresses)
		
		found_addresses = sorted(found_addresses, key=lambda x: x[1], reverse=True)
		return found_addresses[0:3]

	# Function to bypass popups
	def bypass_popup(self):
		xpath_expressions = []
		xpath_expressions.extend([
			'//button[contains(translate(text(),"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"), "accept")]',
			'//button[contains(translate(text(),"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"), "agree")]',
			'//button[contains(translate(text(),"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"), "close")]',
		])
		for xpath in xpath_expressions:
			try:
				a = self.driver.find_element(By.XPATH, xpath)
				a.click()
				break
			except Exception as e:
				pass
		js_script = """
		var elements = document.querySelectorAll('a', 'button');
		for (var i = 0; i < elements.length; i++) {
			for (var j = 0; j < elements[i].attributes.length; j++) {
				var attribute = elements[i].attributes[j];
				if (attribute.value.indexOf('accept') > -1 || attribute.value.indexOf('agree') > -1 || attribute.value.indexOf('close') > -1){
					elements[i].click(); // Perform the click action on the matching element
					return true; // Return true to indicate an element was clicked
				}
			}
		}
		return false; // Return false if no matching element was found to click
		"""
		try:
			self.driver.execute_script(js_script)
		except Exception as e:
			pass
		
	# Function to get the country of the address by testing the website language, the url and the address itself 
	# and if it fails, it uses the zipcodebase api to get the country corresponding to the postcode
	def get_country(self, formatted, adr):
		# Testing uk post code pattern
		if re.findall(r"[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}", formatted['ZIPCODE']):
			formatted['COUNTRY'] = 'UK'
			return formatted
		# Testing if an us state was found in the address
		if formatted['PROVINCE'] in states:
			new = {'COUNTRY': 'US', 'STATE': formatted['PROVINCE']}
			formatted.pop('PROVINCE')
			formatted.pop('COUNTRY')
			new.update(formatted)
			return new
		try:
			ad = re.sub(r'[^\w\s]', '', adr[0]).strip().lower().split()
			state = next(s for i,s in enumerate(ad) if s in states and ad[i+1] == adr[2])
			if state:
				formatted['COUNTRY'] = 'US'
				formatted['STATE'] = state
				return formatted
		except Exception as e:
			pass
		country = None
		# Checking the url domain for the country 
		try:
			country = next(c for c in countries if f".{c}" in self.current_url)
		except Exception as e:
			pass
		#If we didn't find the country in the url, we query the zipcodebase api to get the country corresponding to the postcode
		if country: 
			formatted['COUNTRY'] = country.upper()
			return formatted
		else:
			postcode = formatted['ZIPCODE'].strip()
			headers = { 
			"apikey": "0f6fdb60-da39-11ee-af81-9d2bd0578b49"}

			params = (
			("codes", postcode),
			)
			response = requests.get('https://app.zipcodebase.com/api/v1/search', headers=headers, params=params)
			data = json.loads(response.text)
			if data['results']:
				translator = deepl.Translator('45739d32-4c11-45af-9d1e-06859d3dc182:fx')
				found_city = str(translator.translate_text(formatted['CITY'],target_lang='EN-US')).lower()
				for key in data['results'][f"{postcode}"]:
					try:
						city = key['city'].lower()
						translator = deepl.Translator('45739d32-4c11-45af-9d1e-06859d3dc182:fx')
						city = str(translator.translate_text(city,target_lang='EN-US'))
						if key['country_code'].lower() in countries and (city in found_city or found_city in city):
							formatted['COUNTRY'] = key['country_code']	
					except Exception as e:
						pass
		return formatted
	
	# Function to format the address using the NER model and compute the country code corresponding to the address
	def formatter(self, addresses):
		# we apply the NER model to the top 3 addresses and which one has more labels is the one we choose
		formatted = {}
		lang = None
		try:
			lang = bs(self.page_source, 'html.parser').html.get('lang')
			if 'en' in lang.lower() or address[0][1] < 0.5 or 'us' in lang.lower():
				lang = None
		except Exception as e:
			pass
		max = 0
		ad = None
		for address in addresses:
			f = format_address(str(address[0]).split(), lang)
			length = len([k for k in f.keys() if f[k] != '-'])

			if length > max and (address[1] >= 0.4 or formatted == {}):
				ad = address
				max = length
				formatted = f
				formatted['ZIPCODE'] = re.sub(r'[^\w\s]', '', address[2]).strip()

				formatted['CONFIDENCE'] = address[1]
		if not formatted:
			return None
		return self.get_country(formatted, ad)
		
	# Function to crawl all urls in the urls_list and write addreses to a file in the format: URLS,FORMATTED,TOP CANDIDATES
	# where URLS is the url, FORMATTED is the formatted address and TOP CANDIDATES is the top 3 addresses by probability
	def crawl_to_file(self, file_path):
		mapped_addresses = {}
		with open(file_path, 'w') as f:
			f.write('URLS,FORMATTED,TOP CANDIDATES\n')
		for url in self.urls_list:
			print(f'Loading {url}')
			try:
				response = requests.get(url, timeout=20)
				if response.ok == False:
					self.driver.get(url)
					self.bypass_popup()
					self.page_source = self.driver.page_source
					self.current_url = self.driver.current_url
				else:
					self.page_source = response.text
					self.current_url = url
			except Exception as e:
				with open('logs', 'a') as f:
					f.write(f"Exception occured while loading {url}" + str(e) + '\n')
				continue

			try:
				addresses = self.parse_page2()
				try:
					if addresses[0][1] < 0.9:
						raise Exception('Address not sure')
				except Exception as e:
					other_addresses = self.parse_interesting_urls()
					if other_addresses:
						addresses.extend(other_addresses)
						addresses = sorted(addresses, key=lambda x: x[1], reverse=True)
					if len(addresses) > 3:
						addresses = addresses[0:3]
			except Exception as e:
				with open('logs', 'a') as f:
					f.write(f"Exception occured while parsing {url}" + str(e) + '\n')
				continue

			try:
				with open(file_path, 'a') as f:
					f.write(url + ',')
					if addresses:
						formatted = self.formatter(addresses)
						if formatted:
							f.write('"' + str(formatted).replace('"', '') + '"' + ',')
						else:
							f.write('None,')
					else:
						f.write('None,')
					f.write('"' +str(addresses).replace('"', '') + '"' + '\n')
			except Exception as e:
				with open('logs', 'a') as f:
					f.write(f"Exception occured while writing to file: {url}" + str(e) + '\n\n\n')
		self.driver.quit()
		return mapped_addresses
				
