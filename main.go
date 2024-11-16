package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand/v2"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"syscall"

	"github.com/gorilla/mux"
)

type Peer struct {
	ip   string
	port int
}

type IP_Manager struct {
	mu    sync.Mutex
	peers map[string]Peer
}

type SuperCoolNumber struct {
	mu sync.Mutex
	n  int
}

func newSuperCoolNumber(n int) SuperCoolNumber {
	return SuperCoolNumber{n: n}
}

func newIP_Manager() IP_Manager {
	return IP_Manager{peers: make(map[string]Peer)}
}

var ip_manager IP_Manager
var my_super_cool_number SuperCoolNumber
var client http.Client

func updateModel(n int) {
	my_super_cool_number.mu.Lock()
	defer my_super_cool_number.mu.Unlock()

	my_super_cool_number.n = n
}

func getNRandModels(n int) {
	if len(ip_manager.peers) < n {
		log.Printf("requested %d models but have %d peers", n, len(ip_manager.peers))
		n = len(ip_manager.peers)
	}

	peer_list := make([]string, len(ip_manager.peers))
	models := make([]SuperCoolNumber, n)
	models = append(models, newSuperCoolNumber(0))

	i := 0
	for k := range ip_manager.peers {
		peer_list[i] = k
		i++
	}

	for i := range n {
		i++
		ip := ip_manager.peers[peer_list[rand.IntN(n)]]
		res, err := http.Get("http://" + ip.ip + ":" + strconv.Itoa(ip.port))
		if err != nil {
			log.Println(err.Error())
			continue
		}
		body, err := io.ReadAll(res.Body)
		defer res.Body.Close()

		var a int
		err = json.Unmarshal(body, &a)

		log.Printf("%d: %d", ip.port, a)
	}

}

func shareModel(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(my_super_cool_number.n)
}

func (ip_m *IP_Manager) addPeer(ip string, port int) {
	if ip == "" || port == 0 {
		return
	}
	ip_m.mu.Lock()
	defer ip_m.mu.Unlock()

	ip_m.peers[fmt.Sprintf("%s:%d", ip, port)] = Peer{ip, port}
}

func (ip_m *IP_Manager) getPeerList(w http.ResponseWriter, r *http.Request) {
	ip, _ := getIP(r.RemoteAddr)

	var port struct {
		Port int `json:"port"`
	}
	body, _ := io.ReadAll(r.Body)
	defer r.Body.Close()

	println(string(body))

	err := json.Unmarshal(body, &port)
	if err != nil {
		log.Print("failure unmarshalling address")
		http.Error(w, "malformed port", http.StatusBadRequest)
		return
	}

	log.Printf("Added port %d\n", port)

	if ip != "" {
		ip_m.addPeer(ip, port.Port)
		log.Println(string(r.RemoteAddr))
	}

	keys := make([]string, len(ip_m.peers))

	i := 0
	for key := range ip_m.peers {
		keys[i] = key
		i++
	}

	json.NewEncoder(w).Encode(keys)
}

func (ip_m *IP_Manager) parsePeerList(r *http.Response) error {
	var ips []string

	body, _ := io.ReadAll(r.Body)
	defer r.Body.Close()

	println(body)

	err := json.Unmarshal(body, &ips)
	if err != nil {
		return err
	}

	for _, ip := range ips {
		ip_m.addPeer(getIP(ip))
	}

	return nil
}

// TODO: currently doesnt work for proxies but could extend if wanted
func getIP(s string) (string, int) {
	host, portStr, err := net.SplitHostPort(s)
	if err != nil {
		return "", 0
	}
	port, _ := strconv.Atoi(portStr)
	return host, port
}

func main() {
	my_super_cool_number = newSuperCoolNumber(rand.IntN(100))
	println(my_super_cool_number.n)

	ip_manager = newIP_Manager()
	port, err := strconv.Atoi(os.Args[1])

	if err != nil {
		return
	}

	println(len(os.Args))

	if len(os.Args) > 2 {
		boot_ip := os.Args[2]

		ip_manager.addPeer(getIP(boot_ip))

		var json_body = []byte(fmt.Sprintf(`{"port": %d}`, port))

		res, err := http.Post("http://"+boot_ip+"/getPeerList", "application/json", bytes.NewBuffer(json_body))

		println(res.Status)

		if err != nil {
			log.Printf("Error connecting to boot server: %s\n", err)
		} else {
			err := ip_manager.parsePeerList(res)

			if err != nil {
				log.Printf("Error pasrsing json: %s\n", err)
			} else {
				println("added peers")
				for _, peer := range ip_manager.peers {
					fmt.Printf("     %s: %d\n", peer.ip, peer.port)
				}
			}
		}
	}

	router := mux.NewRouter()
	router.HandleFunc("/getPeerList", ip_manager.getPeerList).Methods("POST")
	router.HandleFunc("/", shareModel).Methods("GET")

	log.Printf("starting server on port %d", port)

	server := &http.Server{
		Addr:    ":" + strconv.Itoa(port),
		Handler: router,
	}

	go func() {
		if err := server.ListenAndServe(); err != nil {
			log.Println(err.Error())
		}
	}()

	getNRandModels(2)

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-sigChan
		fmt.Println("\nShutting down server...")
		server.Close()
		os.Exit(0)
	}()

	select {}

}
