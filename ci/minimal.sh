poetry remove dvc
poetry remove pandas
poetry remove numpy
poetry remove missingno
poetry remove ipykernel --dev

rm -r notebooks
rm -r checklist
rm -r config

rm src/sample.py

rm tests/test_file.py
rm tests/test_sample.py
rm tests/conftest.py
