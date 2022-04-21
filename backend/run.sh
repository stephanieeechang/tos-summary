export PYTHONPATH=/home/sandman/gatech/cs-6471-computational-social-science/project

source /home/sandman/gatech/cs-6471-computational-social-science/project/6471/bin/activate

gunicorn -b 0.0.0.0:5081 -w 3 -t 300 wsgi:app
