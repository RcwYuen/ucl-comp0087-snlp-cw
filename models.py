import torch
from torch.nn.functional import softmax

class Translator:
    def __init__(self, src_tokenizer, tar_tokenizer, model):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tar_tokenizer = tar_tokenizer
        
    def __translate(self, text, fwd, get_probs):
        tokenizer = self.src_tokenizer if fwd else self.tar_tokenizer
        detokenizer = self.tar_tokenizer if fwd else self.src_tokenizer
        langfrom  = self.tar_lang if fwd else self.src_lang

        encoded = tokenizer(text, return_tensors = "pt", padding = True)
        if torch.cuda.is_available():            
            encoded = {k: v.to("cuda") for k, v in encoded.items()}
        
        with torch.no_grad():
            try:
                generated_tokens = self.model.generate(
                    **encoded,
                    forced_bos_token_id = tokenizer.lang_code_to_id[langfrom],
                    output_scores = get_probs,
                    return_dict_in_generate = get_probs
                )
            except AttributeError:
                generated_tokens = self.model.generate(
                    **encoded,
                    output_scores = get_probs,
                    return_dict_in_generate = get_probs
                )
        
        if get_probs:
            probabilities = [softmax(score[0], dim = -1) for score in generated_tokens.scores][1:] # ignoring language token
            generated_ids = generated_tokens.sequences[0][2:-1] # ignoring bos and language token
            token_probs = []
            for i, token_id in enumerate(generated_ids):
                token_probs.append((tokenizer.decode([token_id]), probabilities[i]))
            return token_probs        
        else:
            translated = detokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            return translated
        
    def __alt_word_prob(self, probs, alt, fwd):
        tokenizer = self.tar_tokenizer if fwd else self.src_tokenizer
        alt = [alt.lower()] if isinstance(alt, str) else [i.lower() for i in alt]
        alternative_probabilities = []

        for altword in alt:
            altword_id = tokenizer.encode(altword, add_special_tokens = False)

            for wordid in altword_id:
                altprob = []
                for token, prob in probs:
                    altprob.append((token, prob[wordid].item()))
                alternative_probabilities.append((tokenizer.decode(wordid), altprob))

        return alternative_probabilities

    def forward_translate(self, text):
        return self.__translate(text, True, False)

    def backward_translate(self, text):
        return self.__translate(text, False, False)
    
    def forward_alt_word_probability(self, text, alt):
        probs = self.__translate(text, True, True)
        return self.__alt_word_prob(probs, alt, True)

    def backward_alt_word_probability(self, text, alt):
        probs = self.__translate(text, False, True)
        return self.__alt_word_prob(probs, alt, False)

    def __word_of_interest_known(self, text, in_sentence, replacement, fwd):
        translated = self.forward_translate(text)[0] if fwd else self.backward_translate(text)[0]
        assert in_sentence in translated, "Variable `in_sentence` must contain words from the translated text."
        probs = self.forward_alt_word_probability(text, replacement) \
            if fwd else self.backward_alt_word_probability(text, replacement)
        
        tokenizer = self.tar_tokenizer if fwd else self.src_tokenizer
        in_sentence = tokenizer.encode(in_sentence, add_special_tokens = False)
        tokens = [tokenizer.decode([i]) for i in in_sentence]# ["input_ids"]]
        return {key: {token: val[token] for token in tokens} for key, val in probs.items()}

    def forward_word_of_interest_known(self, text, in_sentence, replacement):
        return self.__word_of_interest_known(text, in_sentence, replacement, True)

    def backward_word_of_interest_known(self, text, in_sentence, replacement):
        return self.__word_of_interest_known(text, in_sentence, replacement, False)

class MBart(Translator):
    def __init__(self, src_lang = "English", tar_lang = "Spanish"):
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        self.src_lang = MBart.get_sym(src_lang.title())
        self.tar_lang = MBart.get_sym(tar_lang.title())
        
        if self.src_lang is None:
            raise ValueError("Source Language is Incorrect")
        if self.tar_lang is None:
            raise ValueError("Target Language is Incorrect")

        self.src_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", \
            src_lang = self.src_lang, tgt_lang = self.tar_lang)
        self.tar_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", \
            src_lang = self.src_lang, tgt_lang = self.tar_lang)
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

        self.src_tokenizer.src_lang = self.src_lang
        self.tar_tokenizer.src_lang = self.tar_lang
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model
        self.model.eval()
        super(MBart, self).__init__(src_tokenizer=self.src_tokenizer, tar_tokenizer=self.tar_tokenizer, model=self.model)

    @staticmethod
    def get_sym(language):
        languages = "Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)".split(", ")
        langdict = {}
        for i in languages:
            i = i.split(" ")
            langdict[i[0]] = i[1].replace("(", "").replace(")", "")
        return langdict[language] if language in langdict.keys() else None

    @staticmethod
    def __str__():
        return "MBart"

class NLLB(Translator):
    def __init__(self, src_lang = None, tar_lang = None):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.src_lang = "eng_Latn" # Now fixated on English -> Spanish
        self.tar_lang = "spa_Latn" # Now fixated on English -> Spanish
        
        if self.src_lang is None:
            raise ValueError("Source Language is Incorrect")
        if self.tar_lang is None:
            raise ValueError("Target Language is Incorrect")

        self.src_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", \
            src_lang = self.src_lang, tgt_lang = self.tar_lang)
        self.tar_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", \
            src_lang = self.src_lang, tgt_lang = self.tar_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

        self.model = self.model.cuda() if torch.cuda.is_available() else self.model
        self.model.eval()
        super(NLLB, self).__init__(src_tokenizer=self.src_tokenizer, tar_tokenizer=self.tar_tokenizer, model=self.model)

    @staticmethod
    def __str__():
        return "NLLB"

class M2M(Translator):
    def __init__(self, src_lang = None, tar_lang = None):
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        self.src_lang = "en" # Now fixated on English -> Spanish
        self.tar_lang = "es" # Now fixated on English -> Spanish
        
        if self.src_lang is None:
            raise ValueError("Source Language is Incorrect")
        if self.tar_lang is None:
            raise ValueError("Target Language is Incorrect")

        self.src_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", \
            src_lang = self.src_lang, tgt_lang = self.tar_lang)
        self.tar_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", \
            src_lang = self.src_lang, tgt_lang = self.tar_lang)
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

        self.src_tokenizer.src_lang = self.src_lang
        self.tar_tokenizer.src_lang = self.tar_lang
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model
        self.model.eval()
        super(M2M, self).__init__(src_tokenizer=self.src_tokenizer, tar_tokenizer=self.tar_tokenizer, model=self.model)

    @staticmethod
    def __str__():
        return "M2M-100"

class MadLad(Translator):
    def __init__(self, src_lang = None, tar_lang = None):
        # https://huggingface.co/google/madlad400-3b-mt
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.src_lang = "en" # Now fixated on English -> Spanish
        self.tar_lang = "es" # Now fixated on English -> Spanish
        
        if self.src_lang is None:
            raise ValueError("Source Language is Incorrect")
        if self.tar_lang is None:
            raise ValueError("Target Language is Incorrect")

        self.src_tokenizer = AutoTokenizer.from_pretrained('jbochi/madlad400-3b-mt', \
            src_lang = self.src_lang, tgt_lang = self.tar_lang)
        self.tar_tokenizer = AutoTokenizer.from_pretrained('jbochi/madlad400-3b-mt', \
            src_lang = self.src_lang, tgt_lang = self.tar_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained('jbochi/madlad400-3b-mt')

        self.src_tokenizer.src_lang = self.src_lang
        self.tar_tokenizer.src_lang = self.tar_lang
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model
        self.model.eval()
        super(MadLad, self).__init__(src_tokenizer=self.src_tokenizer, tar_tokenizer=self.tar_tokenizer, model=self.model)

    @staticmethod
    def __str__():
        return "MADLAD-3b"

class T5Small(Translator):
    def __init__(self, src_lang = None, tar_lang = None):
        # https://huggingface.co/google-t5/t5-small
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.src_lang = "en" # Now fixated on English -> Spanish
        self.tar_lang = "es" # Now fixated on English -> Spanish
        
        if self.src_lang is None:
            raise ValueError("Source Language is Incorrect")
        if self.tar_lang is None:
            raise ValueError("Target Language is Incorrect")

        self.src_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small", \
            src_lang = self.src_lang, tgt_lang = self.tar_lang)
        self.tar_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small", \
            src_lang = self.src_lang, tgt_lang = self.tar_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

        self.src_tokenizer.src_lang = self.src_lang
        self.tar_tokenizer.src_lang = self.tar_lang
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model
        self.model.eval()
        super(T5Small, self).__init__(src_tokenizer=self.src_tokenizer, tar_tokenizer=self.tar_tokenizer, model=self.model)

    @staticmethod
    def __str__():
        return "T5-Small"

class T5Base(Translator):
    def __init__(self, src_lang = None, tar_lang = None):
        # https://huggingface.co/google-t5/t5-small
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.src_lang = "en" # Now fixated on English -> Spanish
        self.tar_lang = "es" # Now fixated on English -> Spanish
        
        if self.src_lang is None:
            raise ValueError("Source Language is Incorrect")
        if self.tar_lang is None:
            raise ValueError("Target Language is Incorrect")

        self.src_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base", \
            src_lang = self.src_lang, tgt_lang = self.tar_lang)
        self.tar_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base", \
            src_lang = self.src_lang, tgt_lang = self.tar_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

        self.src_tokenizer.src_lang = self.src_lang
        self.tar_tokenizer.src_lang = self.tar_lang
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model
        self.model.eval()
        super(T5Base, self).__init__(src_tokenizer=self.src_tokenizer, tar_tokenizer=self.tar_tokenizer, model=self.model)

    @staticmethod
    def __str__():
        return "T5-Base"

class Helsinki(Translator):
    def __init__(self, src_lang = None, tar_lang = None):
        # https://huggingface.co/google-t5/t5-small
        self.src_lang = "English"
        self.tar_lang = "Spanish"
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.src_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
        self.tar_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model
        self.model.eval()
        super(Helsinki, self).__init__(src_tokenizer=self.src_tokenizer, tar_tokenizer=self.tar_tokenizer, model=self.model)

    @staticmethod
    def __str__():
        return "Helsinki"
