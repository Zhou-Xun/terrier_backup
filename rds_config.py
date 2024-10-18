import json

db_config   = {
    "rds_host":"rds-proxy.proxy-cemnk6uitgq7.us-east-2.rds.amazonaws.com",
    "rds_user":"admin",
    "rds_password":"G78F4DN*kMeG*QTRwExj",
    "rds_db":"CLUES_DEV",
    "rds_port":"3306"
    }

db_host = db_config['rds_host']
db_user = db_config['rds_user']
db_password = db_config['rds_password']
db_name = db_config['rds_db']
db_port = db_config['rds_port']