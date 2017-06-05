kill -9 `ps aux | grep $USER\+ | grep kv-store | grep python | awk '{print $2}'`
