import whisper_timestamped as whisper


class Transcriber:
    """The main class to be interacted with"""

    def __init__(self, model="small", audio="", lan="en"):
        self.model = whisper.load_model("medium", device="cpu")
        self.audio = whisper.load_audio(audio)
        self.result = whisper.transcribe(self.model, self.audio, language=lan)
        self.dictionary = self.get_word_dictionary()

    def get_transcript(self):
        """Returns a transcript of words in the song, with no timestamps"""

        return self.result["text"]

    def get_metadata(self):
        """Returns a nested structure containing a transcript of words in the
        song, with their timestamps as well as more accompanying information"""

        return self.result["segments"]

    def get_word_dictionary(self):
        verses = self.get_metadata()

        word_dictionary = {}

        for verse in verses:
            words = verse["words"]

            for word_info in words:
                if "text" in word_info:
                    word = word_info["text"]
                    info = word_info.copy()
                    info.pop("text")

                    if word not in word_dictionary:
                        word_dictionary[word] = []

                    word_dictionary[word].append(info)

        return word_dictionary

    def get_word_data(self, word):
        return self.dictionary[word]

    def get_word_times(self, word):
        info = self.get_word_data(word)
        timestamps = []

        for appearance in info:
            start_time = appearance["start"]
            end_time = appearance["end"]

            word_period = (start_time, end_time)

            timestamps.append(word_period)

        return timestamps


if __name__ == "__main__":
    t = Transcriber(
        audio="/Users/aranyeosakwe/deebo/deebo/data/Melanie Martinez - Play Date (Official Audio).mp3"
    )

    print(t.get_word_dictionary())
