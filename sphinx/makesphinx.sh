# 1.)
echo "1.) create documentation directory"
echo "1.) sphinx-apidoc RootInteractive -o ./docs"
sphinx-apidoc RootInteractive -o ./docs
# 2.)
echo "2.) cp sphinx/conf.py docs/"
cp sphinx/conf.py docs/
# 3.)
echo "3.) make html"
cd docs
make  html
cd ../
# 4.)
echo " 4.) synchronize to server"
echo "rsync -rvzt  docs/_build/* /eos/user/r/rootinteractive/www/"
rsync -rvzt  docs/_build/* /eos/user/r/rootinteractive/www/

