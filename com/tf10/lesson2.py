import tensorflow as tf 

#变量

x = tf.Variable([1, 2])
a = tf.constant([3, 3])

sub = tf.subtract(x, a)

add = tf.add(x, sub)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(sub))
    print(session.run(add))
    
state = tf.Variable(0, name="counter")

new_value = tf.add(state, 1)

update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(state));
    for _ in range(5):
        session.run(update)
        print(session.run(state))
        
        