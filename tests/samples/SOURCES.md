# Sample photos

Public-domain portraits from Wikimedia Commons, used only as test
fixtures for face detection / embedding / clustering verification.

| File | Subject | Source |
|---|---|---|
| einstein_a.jpg | Albert Einstein (1921) | Wikimedia Commons: "Albert Einstein photo 1921.jpg" |
| einstein_b.jpg | Albert Einstein (later portrait) | Wikimedia Commons: "Albert Einstein Head cleaned.jpg" |
| curie_a.jpg | Marie Curie (c. 1920s) | Wikimedia Commons: "Marie Curie c. 1920s.jpg" |
| landscape.jpg | Synthetic landscape (no face) | Generated in-repo with Pillow |

Expected results: one face in each portrait, none in the landscape;
the two Einstein photos cluster into one identity, Curie forms her
own (same-person cosine ~0.35 for this cross-age pair with the small
ArcFace model; different-person <= ~0.15).
