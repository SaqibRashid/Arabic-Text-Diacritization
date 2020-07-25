# Arabic-Text-Diacritization

The Arabic language is considered among the most widely spoken languages in the world; approximately half a billion people speak it in twenty-six countries. Modern Standard Arabic (MSA) is used in newspapers, books, and the Internet. MSA is typically written without diacritics because native speakers can understand the context in which the word is used, hence, they read the word in its correct form. However, non-natives require diacritics to judge the correct form of a word. Diacritics are markings that indicate a difference in the pronunciation of a word. In a language like Arabic, these markings especially help in pronouncing and understanding the meaning of the word.

## Goal of the project
Students of Arabic who learn it as a foreign language first develop their understanding of a marked Arabic text by applying grammatical rules. As a step forward, most of these non-natives, especially the religious community, aim to learn how to read a classical text. This class of texts is very large in number but is found to be unmarked in printed and digital form. To practice reading a classical text as a beginner, a marked sample is required. Successfully implementing this project we are trying to help such students by opening up a large number of books which they can read and practice their reading skills.

## Implementation
Our training data is comprised of 3615 Diacritized Arabic sentences. We preprocess them (data encoding) and divide them into input and target sequences of size 5 in order to train over model. Our model contains Bidirectional Long-Short Term Memory units with RelU activation. Due to limitated computational power we kept our data size and number of epochs as low as possible. Even then our system after post processing is giving us 80% accuracy on unseen data. 

*See the Final presentation Powerpoint file for further details*
