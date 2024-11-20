package main

import (
	"context"
	pb "dist_final_project/proto"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/peer"
)

var (
	port       = flag.Uint64("port", 8080, "server port")
	boot_ip    = flag.String("boo_ip", "localhost:8080", "boot server ip")
	ip_manager = IP_Manager{Peers: make(map[string]*pb.Peer)}
)

type server struct {
	pb.UnimplementedModelServiceServer
}

type IP_Manager struct {
	mu    sync.RWMutex
	Peers map[string]*pb.Peer // should make more efficient struct??
}

func (ip_manager *IP_Manager) AddPeer(ctx context.Context, port uint32) error {
	p, ok := peer.FromContext(ctx)
	if !ok {
		return errors.New("could not get peer info from context")
	}

	addr := p.Addr.String()
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return err
	}
	if host == "::1" {
		host = "localhost"
	}

	ip_manager.mu.Lock()
	defer ip_manager.mu.Unlock()

	ip_manager.Peers[host] = &pb.Peer{Ip: host, Port: port}

	return nil
}

// NOTE: this is creating a copy mainly for thread safety, not sure how necessary it is but better to be safe at this stage
func (ip_manager *IP_Manager) GetPeerList() map[string]*pb.Peer {
	ip_manager.mu.RLock()
	defer ip_manager.mu.RUnlock()

	peers := make(map[string]*pb.Peer, len(ip_manager.Peers))

	for k, v := range ip_manager.Peers {
		peers[k] = v
	}

	return peers
}

func (s *server) GetModel(in *pb.GetModelRequest, stream pb.ModelService_GetModelServer) error {
	file, err := os.Open("/Users/nathanvosburg/Documents/CS339/dist_final_project/models/model1.pth")
	if err != nil {
		return err
	}
	defer file.Close()

	buf := make([]byte, 1024*64)
	batch_number := 0
	for {
		n, err := file.Read(buf)
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		err = stream.Send(&pb.GetModelReply{Chunk: buf[:n]})
		if err != nil {
			return err
		}
		log.Printf("Sent - batch #%v - size - %v\n", batch_number, n)
		batch_number += 1
	}

	return nil
}

func (s *server) GetPeerList(ctx context.Context, in *pb.GetPeerListRequest) (*pb.GetPeerListResponse, error) {
	err := ip_manager.AddPeer(ctx, in.Port)
	if err != nil {
		log.Printf("Error getting adding peer from context: %s", err)
	}

	log.Printf("Received: %v", in.Port)
	return &pb.GetPeerListResponse{
		Peers: ip_manager.Peers,
	}, nil
}

func getRandomModel(id uint32, peers []pb.Peer) error {
	file, err := os.Create(fmt.Sprintf("model%d.pth", id))
	if err != nil {
		return fmt.Errorf("failed to create file: %s", err.Error())
	}
	defer file.Close()

	conn, err := grpc.NewClient("localhost:8080", grpc.WithInsecure())
	if err != nil {
		return fmt.Errorf("failed to connect: %s", err.Error())
	}
	defer conn.Close()

	client := pb.NewModelServiceClient(conn)

	stream, err := client.GetModel(context.Background(), &pb.GetModelRequest{Port: uint32(*port)})
	if err != nil {
		return fmt.Errorf("error getting model: %s", err.Error())
	}

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to recieve chunk: %s", err.Error())
		}

		_, err = file.Write(chunk.Chunk)
		if err != nil {
			return fmt.Errorf("failed to write chunk: %s", err.Error())
		}
	}
	log.Println("Model download complete!")

	return nil
}

func (s *server) CollectModels(_ context.Context, in *pb.CollectModelRequest) (*pb.CollectModelResponse, error) {
	if in.Key != "key" {
		return &pb.CollectModelResponse{
			Success: false,
		}, errors.New("unauthorized")
	}

	//n := int(in.GetNum())

	return &pb.CollectModelResponse{
		Success: true,
	}, nil
}

func runServer() {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	pb.RegisterModelServiceServer(s, &server{})

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
		<-sigCh
		s.GracefulStop()
	}()

	log.Printf("server listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

func boot() error {
	conn, err := grpc.NewClient(*boot_ip, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	defer conn.Close()
	c := pb.NewModelServiceClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	r, err := c.GetPeerList(ctx, &pb.GetPeerListRequest{Port: uint32(*port)})
	if err != nil {
		return err
	}

	err = getRandomModel(1, nil)
	if err != nil {
		return err
	}

	m := r.GetPeers()

	ip_manager.mu.Lock()
	ip_manager.Peers = m
	ip_manager.mu.Unlock()

	for _, v := range m {
		fmt.Printf("%s:%d\n", v.Ip, v.Port)
	}

	return nil
}

func main() {
	flag.Parse()

	if err := boot(); err != nil {
		log.Printf("Error bootsraping: %s", err.Error())
	}

	runServer()
}
