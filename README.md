# bayrepo_neuro

Build

gcc -o web_test3 neuro_web.c neuro.c libcivetweb.a -lpthread -lgsl -lm -lpng -lcrypto

Depends on

https://github.com/civetweb/civetweb
Build civetweb with params: make -j2 COPT='-DNO_SSL -DNO_SSL_DL' lib

neuro_web_client.c should be linked with -lcurl

Last build

cmake .
make


Usage example: https://brepo.ru/post/eksperimenty-s-neirosetyami
