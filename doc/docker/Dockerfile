FROM httpd:2.4

LABEL maintainer="Jeff Zaremba <jzaremba@gmail.com>"

COPY ./httpd.conf /usr/local/apache2/conf/httpd.conf

RUN chown www-data:www-data /usr/local/apache2/conf/httpd.conf

VOLUME ["/usr/local/apache2/htdocs/"]
