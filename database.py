import pymysql
# from detect_pipeline import FaceRecognizePipeline
# from PIL import Image


class Database():
    def __init__(self, host='127.0.0.1', user='root', password='123456', db='face'):
        self.host = host
        self.user = user
        self.password = password
        self.db = db
        self.conn = None
        self.cursor = None

    def connect_database(self):
        try:
            self.conn = pymysql.connect(host=self.host, user=self.user, password=self.password, db=self.db, port=3306)
        except pymysql.Error as e:
            print(e)
            return False
        self.cursor = self.conn.cursor()
        return True

    def close_database(self):
        if self.conn and self.cursor:
            self.cursor.close()
            self.conn.close()
        return True

    def function(self, sql, msg=None):
        flag = False
        if self.connect_database():
            try:
                self.cursor.execute(sql)
                self.conn.commit()
                flag = True
                if msg is not None:
                    print(msg)
            except pymysql.Error as e:
                print(e)
                self.conn.rollback()
            finally:
                self.close_database()
        return flag

    def create_table(self, table_name='facebank'):
        self.connect_database()
        if self.conn and self.cursor:
            self.cursor.execute("DROP TABLE IF EXISTS %s" % table_name)
            sql = """CREATE TABLE facebank(
                     id int unsigned not null auto_increment,
                     name varchar(50) not null,
                     picname varchar(100),
                     embedding text,
                     primary key(id)
            )
            """
            try:
                self.cursor.execute(sql)
                self.conn.commit()
                print("建表成功！")
            except:
                self.conn.rollback()
            finally:
                self.close_database()
        return True

    def update_data(self, name, embedding, pic_name=None):
        if pic_name is not None:
            sql = "update facebank set embedding='%s' where name='%s';" % (embedding, name)
        else:
            sql = "update facebank set embedding='%s' where name='%s' and picname='%s';" % (embedding, name, pic_name)
        return self.function(sql, '修改成功！')

    def add_data(self, name, embedding, pic_name=None):
        if pic_name is not None:
            success, _ = self.search_data(name, pic_name)
            if success:
                return self.update_data(name, embedding, pic_name)
            else:
                sql = '''insert into facebank(name,
                        picname, embedding) values 
                        ('%s', '%s', '%s') ''' % (name, pic_name, embedding)
        else:
            sql = '''insert into facebank(name,
                    embedding) values 
                    ('%s', '%s') ''' % (name, embedding)
        return self.function(sql, '插入成功!')

    def search_all(self):
        sql = 'select * from facebank;'
        self.function(sql, '查找成功！')
        result = self.cursor.fetchall()
        if len(result) > 0:
            return True, result
        else:
            return False, result

    def search_data(self, name=None, pic_name=None):
        if name is not None and pic_name is not  None:
            sql = "select * from facebank where name = '%s' and picname = '%s';" % (name, pic_name)
        elif name is not None:
            sql = "select * from facebank where name = '%s';" % name
        elif pic_name is not None:
            sql = "select * from facebank where picname = '%s';" % pic_name
        else:
            return self.search_all()
        self.function(sql, '查找成功！')
        result = self.cursor.fetchall()
        if len(result) > 0:
            return True, result
        else:
            return False, result

    def delete_data(self, name=None, pic_name=None):
        if name is not None and pic_name is not None:
            sql = "delete from facebank where name = '%s' and picname = '%s';" % (name, pic_name)
        elif name is not None:
            sql = "delete from facebank where name = '%s';" % name
        elif pic_name is not None:
            sql = "delete from facebank where picname = '%s';" % pic_name
        else:
            return True
        return self.function(sql, '删除成功！')


if __name__ == '__main__':

    db = Database()
    # db.create_table()
    # facedetect = FaceRecognizePipeline()
    # img = Image.open('./data/facebank/wsl/wsl_1.jpg')
    # emb = facedetect.img2embeddings(img)
    # name = 'wsl'
    # pic_name = 'wsl_1.jpg'
    # emb_str = ','.join([str(i) for i in emb[0]])
    # db.add_data(name=name, pic_name=pic_name, embedding=emb_str)
    # db.update_data(name, emb)
    _, result = db.search_data()
    print(len(result), result)