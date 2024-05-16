#include <iostream>
#include <fstream>
#include <queue>
#include <future>
#include <list>
#include <thread>
#include <chrono>
#include <cmath>
#include <functional>
#include <mutex>
#include <random>
#include <condition_variable>

template <typename T>

class Server{
    private:
    std::thread thread_of_server;
    std::mutex mut;
    std::queue<std::pair<size_t, std::future<T>>> tasks;
    std::unordered_map<size_t, T> results;
    std::condition_variable cv;
    size_t id = 1;
    bool flag = true;
    public: 


void start()
{
    flag = false;
    thread_of_server = std::thread(&Server::server_thread, this);
    std::cout<< "server start" << std::endl;
}

void stop()
{
    {std::unique_lock<std::mutex> lock_res(mut);
    flag = true;
    cv.notify_one();
    }
    thread_of_server.join();
}

T request_result(size_t id_res)
{
    std::unique_lock<std::mutex> lock_res(mut);
    cv.wait(lock_res, [this, id_res]() {return results.find(id_res) != results.end();});
    T result = results[id_res];
    results.erase(id_res);
    return result;
}

void server_thread()
{
    
    while (true)
{
        std::unique_lock<std::mutex> lock_res(mut);
        size_t id_task;
        cv.wait(lock_res, [this]() {return !tasks.empty() || flag;});
        if (tasks.empty() && flag)
            break;
        if (!tasks.empty())
        {
            auto task = std::move(tasks.front());
            tasks.pop();
            results[task.first] = task.second.get();
            cv.notify_one();
        }
    }

    std::cout << "Server stop!\n";
}

size_t add_task(std::function<T()> task)
{
    // id задачи
    size_t id_task = id;
    id++;

    // блокировщик для работы с общими данными
    std::unique_lock<std::mutex> lock_res(mut);

    // создаем задачу (ленивое выполнение)
    tasks.push({id_task, std::async(std::launch::deferred, task)});
    cv.notify_one();
    return tasks.back().first;
}

};

template<typename T>

class Client {
    public:
        std::vector<std::pair<int, T>> task_id;
        std::list<std::pair<T,T>> client_res (Server <T>& server)
        {
            std::list<std::pair<T,T>> results;
            for (const auto& pair : task_id)
            {
                T result = server.request_result(pair.first);
                results.push_back({pair.second, result});
            }
            return results;
        }
        void run (Server<T>& server, std::function<std::pair<T,T>()> gen_task)
        {
            auto task = gen_task();
            int id = server.add_task([task]() {return task.second; });
            task_id.push_back({id, task.first});
        }

};

template<typename T> 
std::pair<T,T> fsinus()
{
    static std::default_random_engine gen;
    static std::uniform_real_distribution<T> distr (-3.14159, 3.14159);
    T x = distr(gen);
    return {x, std::sin(x)};
}

template<typename T> 
std::pair<T,T> fsq()
{
    static std::default_random_engine gen;
    static std::uniform_real_distribution<T> distr (1.0, 10.0);
    T x = distr(gen);
    return {x, std::sqrt(x)};
}

template<typename T> 
std::pair<T,T> fpow()
{
    static std::default_random_engine gen;
    static std::uniform_real_distribution<T> distr (1.0, 10.0);
    T x = distr(gen);
    return {x, std::pow(x, 2.0)};
}


int main()
{
    Server<double> server;
    server.start();
    Client<double> cl1;
    Client<double> cl2;
    Client<double> cl3;

    for (size_t i = 0; i < 10000; i++)
    {
        cl1.run(server, fsinus<double>);
        cl2.run(server, fsq<double>);
        cl3.run(server, fpow<double>);
    }
    
    std::list<std::pair<double, double>> thread1 = cl1.client_res(server);
    std::list<std::pair<double, double>> thread2 = cl2.client_res(server);
    std::list<std::pair<double, double>> thread3 = cl3.client_res(server);

    server.stop();

    std::ofstream test1("test1.txt");
    std::ofstream test2("test2.txt");
    std::ofstream test3("test3.txt");

    for (const auto& p:thread1)
        test1 << "sinus ( " << p.first << " ) = " << p.second << std::endl;
    test1.close();

    for (const auto& p:thread2)
        test2 << "sqrt ( " << p.first << " ) = " << p.second << std::endl;
    test2.close();

    for (const auto& p:thread3)
        test3 << "pow ( " << p.first << " ) = " << p.second << std::endl;
    test3.close();

return 0;
}