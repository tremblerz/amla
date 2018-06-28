#####################
Managing DNS entries
#####################

Managing DNS is super easy for the System Administrators. Our primary DNS entry for web portals
of institutes is **iiits.in**. For example `cosmos.iiits.in <http://cosmos.iiits.in>`__ , `myspace.iiits.in <http://myspace.iiits.in>`__ , `studyspace.iiits.in <http://studyspace.iiits.in>`__ 

Steps to follow
-----------------

* Log In to DNS Server. IP address of DNS server at time of writing documentation is ``10.0.1.2``. Server can be logged in by doing SSH to primary server sitting at ``10.0.1.29`` and then opening the appropriate container.
* Open the *iiits.in.forward* file located at ``/var/named/iiits.in.forward`` using some editor.\
* You will find many DNS entries already present out there in the file. Entry will be in the following format ``[secondary-domain-name] [IN] [record-type] [IP-address]``.\
* Let's say you want to add **test.iiits.in** mapping to IP **10.0.3.33**. Then the corresponding entry would be ``test IN A 10.0.3.33``.\