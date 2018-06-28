##################
Working for Cosmos
##################

`Cosmos <http://cosmos.iiits.in>`__ is the Intranet portal for IIITS where all of the required web portals/apps needed by students,
staff and administrators reside.


Setting up Cosmos locally
---------------------------
First thing to begin with, for contributing to Cosmos is setting up its local copy. This step is very important for the developer to make changes and see the results locally. Cosmos code is hosted with the help of Github at `Cosmos code <https://github.com/IIITS/cosmos-2.0>`__. Following steps need to be followed to set up the environment.

1. Make a new folder and set up a Python virtual environment.
2. Clone the code from repository by typing command ``git clone https://github.com/IIITS/cosmos-2.0.git``.
3. Create your topic branch by giving the command ``git checkout -b branch-name``.
4. Once the code is cloned and topic branch is created go to folder *cosmos-2.0* and type ``pip install requirements.txt``.
5. After all dependencies are installed then type ``./manage.py makemigrations`` which will make files for SQL migrations and now give command ``./manage.py migrate`` after which tables will be created in your database locally.
6. Now under folder *cosmos* there will be a file *settings.py* which needs to be tweaked for static files configurations. This might be a very irritating process for the ones who don't have idea about Django's staticfiles since it has not been managed well. Please follow next step.
7. So, idea is to have all the staticfiles to be seen by your Django server. In view of this we need to run command ``./manage.py collectstatic``. But running this command will give error for the static root path. My approach to solve this problem was first commenting out ``STATICFILES_DIRS`` variable in *settings.py* and then running the command and once static files are migrated then uncommenting the line ``#os.path.join(BASE_DIR, 'staticfiles'),``
8. After tweaking at this point once you are able to run ``./manage.py collectstatic`` successfully then launch your development server by doing ``./manage.py runserver``.

Making changes and setting up new portals
------------------------------------------
After you are able to run cosmos application at your local machine you can start writing new code or making changes but you have to adhere to few things(NOTE: Please have a good knowledge of Django to understand few terms).

1. Make use of base template and nav-bars available in ``templates`` directory.
2. Static files for all app must reside in the folder dedicated to them inside ``staticfiles`` directory.
3. Before you make any change in the code you have to create your own branch as described in first section also.

Pushing changes on production server
-------------------------------------

1. Once you've make the suitable changes and tested enough you can push the changes of your branch.
2. After pushing the changes you can create a pull request through Github and merge the request if you have neccessary permissions.
3. Now, the overall changes has been made you need to bring it into the production server.
4. For this you need to ``git fetch origin master`` and then apply ``git merge origin/master`` to merge the changes locally. Finally you may run ``./manage.py makemigrations && ./manage.py migrate`` for updating the tables.
5. Once all this is done, you need to restart the apache server by issuing command ``sudo systemctl httpd restart``
