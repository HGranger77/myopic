
resource management for a home server
different topics of interest for each user

locational data / map / story geotagging
fact checking
enhanced sentiment analysis
  - emotion detection (7-label: anger/disgust/fear/joy/neutral/sadness/surprise) and
    per-entity sentiment are both built and running, but the emotion badge is disabled
    in the UI for now - needs a better presentation than a raw 7-way label before it's
    worth showing again
  - underlying agenda/bias detection (does an article have a hidden agenda - political
    promotion, tourism/advertorial framing, etc): tried bucketresearch/politicalBiasBERT,
    failed validation hard - inverted on a deliberately unambiguous right-leaning example,
    gave a confident-looking LEFT label on travel-promo text with no political content at
    all. Not implemented. Worth revisiting with a propaganda-technique classifier (trained
    on the SemEval PTC dataset - loaded language/flag waving/appeal to fear etc, rather than
    a single left/right axis) if this is still wanted later, but not for the current MR
translation and foreign news sources

seperate by day
map view
logo
