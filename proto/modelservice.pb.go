// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.35.2
// 	protoc        v3.20.3
// source: proto/modelservice.proto

package proto

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type CollectModelsRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Key string `protobuf:"bytes,1,opt,name=key,proto3" json:"key,omitempty"`
	Num uint32 `protobuf:"varint,2,opt,name=num,proto3" json:"num,omitempty"`
}

func (x *CollectModelsRequest) Reset() {
	*x = CollectModelsRequest{}
	mi := &file_proto_modelservice_proto_msgTypes[0]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CollectModelsRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CollectModelsRequest) ProtoMessage() {}

func (x *CollectModelsRequest) ProtoReflect() protoreflect.Message {
	mi := &file_proto_modelservice_proto_msgTypes[0]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CollectModelsRequest.ProtoReflect.Descriptor instead.
func (*CollectModelsRequest) Descriptor() ([]byte, []int) {
	return file_proto_modelservice_proto_rawDescGZIP(), []int{0}
}

func (x *CollectModelsRequest) GetKey() string {
	if x != nil {
		return x.Key
	}
	return ""
}

func (x *CollectModelsRequest) GetNum() uint32 {
	if x != nil {
		return x.Num
	}
	return 0
}

type CollectModelsResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Success bool `protobuf:"varint,1,opt,name=success,proto3" json:"success,omitempty"`
}

func (x *CollectModelsResponse) Reset() {
	*x = CollectModelsResponse{}
	mi := &file_proto_modelservice_proto_msgTypes[1]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *CollectModelsResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*CollectModelsResponse) ProtoMessage() {}

func (x *CollectModelsResponse) ProtoReflect() protoreflect.Message {
	mi := &file_proto_modelservice_proto_msgTypes[1]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use CollectModelsResponse.ProtoReflect.Descriptor instead.
func (*CollectModelsResponse) Descriptor() ([]byte, []int) {
	return file_proto_modelservice_proto_rawDescGZIP(), []int{1}
}

func (x *CollectModelsResponse) GetSuccess() bool {
	if x != nil {
		return x.Success
	}
	return false
}

type GetModelRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Port uint32 `protobuf:"varint,1,opt,name=port,proto3" json:"port,omitempty"`
}

func (x *GetModelRequest) Reset() {
	*x = GetModelRequest{}
	mi := &file_proto_modelservice_proto_msgTypes[2]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *GetModelRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GetModelRequest) ProtoMessage() {}

func (x *GetModelRequest) ProtoReflect() protoreflect.Message {
	mi := &file_proto_modelservice_proto_msgTypes[2]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GetModelRequest.ProtoReflect.Descriptor instead.
func (*GetModelRequest) Descriptor() ([]byte, []int) {
	return file_proto_modelservice_proto_rawDescGZIP(), []int{2}
}

func (x *GetModelRequest) GetPort() uint32 {
	if x != nil {
		return x.Port
	}
	return 0
}

type GetModelReply struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Chunk []byte `protobuf:"bytes,1,opt,name=chunk,proto3" json:"chunk,omitempty"`
}

func (x *GetModelReply) Reset() {
	*x = GetModelReply{}
	mi := &file_proto_modelservice_proto_msgTypes[3]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *GetModelReply) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GetModelReply) ProtoMessage() {}

func (x *GetModelReply) ProtoReflect() protoreflect.Message {
	mi := &file_proto_modelservice_proto_msgTypes[3]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GetModelReply.ProtoReflect.Descriptor instead.
func (*GetModelReply) Descriptor() ([]byte, []int) {
	return file_proto_modelservice_proto_rawDescGZIP(), []int{3}
}

func (x *GetModelReply) GetChunk() []byte {
	if x != nil {
		return x.Chunk
	}
	return nil
}

type GetBootModelRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Port uint32 `protobuf:"varint,1,opt,name=port,proto3" json:"port,omitempty"`
}

func (x *GetBootModelRequest) Reset() {
	*x = GetBootModelRequest{}
	mi := &file_proto_modelservice_proto_msgTypes[4]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *GetBootModelRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GetBootModelRequest) ProtoMessage() {}

func (x *GetBootModelRequest) ProtoReflect() protoreflect.Message {
	mi := &file_proto_modelservice_proto_msgTypes[4]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GetBootModelRequest.ProtoReflect.Descriptor instead.
func (*GetBootModelRequest) Descriptor() ([]byte, []int) {
	return file_proto_modelservice_proto_rawDescGZIP(), []int{4}
}

func (x *GetBootModelRequest) GetPort() uint32 {
	if x != nil {
		return x.Port
	}
	return 0
}

type GetBootModelReply struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Chunk []byte `protobuf:"bytes,1,opt,name=chunk,proto3" json:"chunk,omitempty"`
}

func (x *GetBootModelReply) Reset() {
	*x = GetBootModelReply{}
	mi := &file_proto_modelservice_proto_msgTypes[5]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *GetBootModelReply) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GetBootModelReply) ProtoMessage() {}

func (x *GetBootModelReply) ProtoReflect() protoreflect.Message {
	mi := &file_proto_modelservice_proto_msgTypes[5]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GetBootModelReply.ProtoReflect.Descriptor instead.
func (*GetBootModelReply) Descriptor() ([]byte, []int) {
	return file_proto_modelservice_proto_rawDescGZIP(), []int{5}
}

func (x *GetBootModelReply) GetChunk() []byte {
	if x != nil {
		return x.Chunk
	}
	return nil
}

type GetPeerListRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Port uint32 `protobuf:"varint,1,opt,name=port,proto3" json:"port,omitempty"`
}

func (x *GetPeerListRequest) Reset() {
	*x = GetPeerListRequest{}
	mi := &file_proto_modelservice_proto_msgTypes[6]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *GetPeerListRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GetPeerListRequest) ProtoMessage() {}

func (x *GetPeerListRequest) ProtoReflect() protoreflect.Message {
	mi := &file_proto_modelservice_proto_msgTypes[6]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GetPeerListRequest.ProtoReflect.Descriptor instead.
func (*GetPeerListRequest) Descriptor() ([]byte, []int) {
	return file_proto_modelservice_proto_rawDescGZIP(), []int{6}
}

func (x *GetPeerListRequest) GetPort() uint32 {
	if x != nil {
		return x.Port
	}
	return 0
}

type GetPeerListResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Peers map[string]*Peer `protobuf:"bytes,1,rep,name=peers,proto3" json:"peers,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
}

func (x *GetPeerListResponse) Reset() {
	*x = GetPeerListResponse{}
	mi := &file_proto_modelservice_proto_msgTypes[7]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *GetPeerListResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GetPeerListResponse) ProtoMessage() {}

func (x *GetPeerListResponse) ProtoReflect() protoreflect.Message {
	mi := &file_proto_modelservice_proto_msgTypes[7]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GetPeerListResponse.ProtoReflect.Descriptor instead.
func (*GetPeerListResponse) Descriptor() ([]byte, []int) {
	return file_proto_modelservice_proto_rawDescGZIP(), []int{7}
}

func (x *GetPeerListResponse) GetPeers() map[string]*Peer {
	if x != nil {
		return x.Peers
	}
	return nil
}

type Peer struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Ip   string `protobuf:"bytes,1,opt,name=ip,proto3" json:"ip,omitempty"`
	Port uint32 `protobuf:"varint,2,opt,name=port,proto3" json:"port,omitempty"`
}

func (x *Peer) Reset() {
	*x = Peer{}
	mi := &file_proto_modelservice_proto_msgTypes[8]
	ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
	ms.StoreMessageInfo(mi)
}

func (x *Peer) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Peer) ProtoMessage() {}

func (x *Peer) ProtoReflect() protoreflect.Message {
	mi := &file_proto_modelservice_proto_msgTypes[8]
	if x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Peer.ProtoReflect.Descriptor instead.
func (*Peer) Descriptor() ([]byte, []int) {
	return file_proto_modelservice_proto_rawDescGZIP(), []int{8}
}

func (x *Peer) GetIp() string {
	if x != nil {
		return x.Ip
	}
	return ""
}

func (x *Peer) GetPort() uint32 {
	if x != nil {
		return x.Port
	}
	return 0
}

var File_proto_modelservice_proto protoreflect.FileDescriptor

var file_proto_modelservice_proto_rawDesc = []byte{
	0x0a, 0x18, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2f, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x73, 0x65, 0x72,
	0x76, 0x69, 0x63, 0x65, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x0c, 0x6d, 0x6f, 0x64, 0x65,
	0x6c, 0x73, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x22, 0x3a, 0x0a, 0x14, 0x43, 0x6f, 0x6c, 0x6c,
	0x65, 0x63, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x73, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74,
	0x12, 0x10, 0x0a, 0x03, 0x6b, 0x65, 0x79, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x03, 0x6b,
	0x65, 0x79, 0x12, 0x10, 0x0a, 0x03, 0x6e, 0x75, 0x6d, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0d, 0x52,
	0x03, 0x6e, 0x75, 0x6d, 0x22, 0x31, 0x0a, 0x15, 0x43, 0x6f, 0x6c, 0x6c, 0x65, 0x63, 0x74, 0x4d,
	0x6f, 0x64, 0x65, 0x6c, 0x73, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x18, 0x0a,
	0x07, 0x73, 0x75, 0x63, 0x63, 0x65, 0x73, 0x73, 0x18, 0x01, 0x20, 0x01, 0x28, 0x08, 0x52, 0x07,
	0x73, 0x75, 0x63, 0x63, 0x65, 0x73, 0x73, 0x22, 0x25, 0x0a, 0x0f, 0x47, 0x65, 0x74, 0x4d, 0x6f,
	0x64, 0x65, 0x6c, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x12, 0x0a, 0x04, 0x70, 0x6f,
	0x72, 0x74, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x04, 0x70, 0x6f, 0x72, 0x74, 0x22, 0x25,
	0x0a, 0x0d, 0x47, 0x65, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x52, 0x65, 0x70, 0x6c, 0x79, 0x12,
	0x14, 0x0a, 0x05, 0x63, 0x68, 0x75, 0x6e, 0x6b, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0c, 0x52, 0x05,
	0x63, 0x68, 0x75, 0x6e, 0x6b, 0x22, 0x29, 0x0a, 0x13, 0x47, 0x65, 0x74, 0x42, 0x6f, 0x6f, 0x74,
	0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x12, 0x0a, 0x04,
	0x70, 0x6f, 0x72, 0x74, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x04, 0x70, 0x6f, 0x72, 0x74,
	0x22, 0x29, 0x0a, 0x11, 0x47, 0x65, 0x74, 0x42, 0x6f, 0x6f, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c,
	0x52, 0x65, 0x70, 0x6c, 0x79, 0x12, 0x14, 0x0a, 0x05, 0x63, 0x68, 0x75, 0x6e, 0x6b, 0x18, 0x01,
	0x20, 0x01, 0x28, 0x0c, 0x52, 0x05, 0x63, 0x68, 0x75, 0x6e, 0x6b, 0x22, 0x28, 0x0a, 0x12, 0x47,
	0x65, 0x74, 0x50, 0x65, 0x65, 0x72, 0x4c, 0x69, 0x73, 0x74, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73,
	0x74, 0x12, 0x12, 0x0a, 0x04, 0x70, 0x6f, 0x72, 0x74, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0d, 0x52,
	0x04, 0x70, 0x6f, 0x72, 0x74, 0x22, 0xa7, 0x01, 0x0a, 0x13, 0x47, 0x65, 0x74, 0x50, 0x65, 0x65,
	0x72, 0x4c, 0x69, 0x73, 0x74, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x42, 0x0a,
	0x05, 0x70, 0x65, 0x65, 0x72, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x2c, 0x2e, 0x6d,
	0x6f, 0x64, 0x65, 0x6c, 0x73, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x2e, 0x47, 0x65, 0x74, 0x50,
	0x65, 0x65, 0x72, 0x4c, 0x69, 0x73, 0x74, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x2e,
	0x50, 0x65, 0x65, 0x72, 0x73, 0x45, 0x6e, 0x74, 0x72, 0x79, 0x52, 0x05, 0x70, 0x65, 0x65, 0x72,
	0x73, 0x1a, 0x4c, 0x0a, 0x0a, 0x50, 0x65, 0x65, 0x72, 0x73, 0x45, 0x6e, 0x74, 0x72, 0x79, 0x12,
	0x10, 0x0a, 0x03, 0x6b, 0x65, 0x79, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x03, 0x6b, 0x65,
	0x79, 0x12, 0x28, 0x0a, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b,
	0x32, 0x12, 0x2e, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x73, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x2e,
	0x50, 0x65, 0x65, 0x72, 0x52, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x3a, 0x02, 0x38, 0x01, 0x22,
	0x2a, 0x0a, 0x04, 0x50, 0x65, 0x65, 0x72, 0x12, 0x0e, 0x0a, 0x02, 0x69, 0x70, 0x18, 0x01, 0x20,
	0x01, 0x28, 0x09, 0x52, 0x02, 0x69, 0x70, 0x12, 0x12, 0x0a, 0x04, 0x70, 0x6f, 0x72, 0x74, 0x18,
	0x02, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x04, 0x70, 0x6f, 0x72, 0x74, 0x32, 0xe4, 0x02, 0x0a, 0x0c,
	0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x53, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x12, 0x4a, 0x0a, 0x08,
	0x47, 0x65, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x12, 0x1d, 0x2e, 0x6d, 0x6f, 0x64, 0x65, 0x6c,
	0x73, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x2e, 0x47, 0x65, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c,
	0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1b, 0x2e, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x73,
	0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x2e, 0x47, 0x65, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x52,
	0x65, 0x70, 0x6c, 0x79, 0x22, 0x00, 0x30, 0x01, 0x12, 0x54, 0x0a, 0x0b, 0x47, 0x65, 0x74, 0x50,
	0x65, 0x65, 0x72, 0x4c, 0x69, 0x73, 0x74, 0x12, 0x20, 0x2e, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x73,
	0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x2e, 0x47, 0x65, 0x74, 0x50, 0x65, 0x65, 0x72, 0x4c, 0x69,
	0x73, 0x74, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x21, 0x2e, 0x6d, 0x6f, 0x64, 0x65,
	0x6c, 0x73, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x2e, 0x47, 0x65, 0x74, 0x50, 0x65, 0x65, 0x72,
	0x4c, 0x69, 0x73, 0x74, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x12, 0x5a,
	0x0a, 0x0d, 0x43, 0x6f, 0x6c, 0x6c, 0x65, 0x63, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x73, 0x12,
	0x22, 0x2e, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x73, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x2e, 0x43,
	0x6f, 0x6c, 0x6c, 0x65, 0x63, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x73, 0x52, 0x65, 0x71, 0x75,
	0x65, 0x73, 0x74, 0x1a, 0x23, 0x2e, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x73, 0x65, 0x72, 0x76, 0x69,
	0x63, 0x65, 0x2e, 0x43, 0x6f, 0x6c, 0x6c, 0x65, 0x63, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x73,
	0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x22, 0x00, 0x12, 0x56, 0x0a, 0x0c, 0x47, 0x65,
	0x74, 0x42, 0x6f, 0x6f, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x12, 0x21, 0x2e, 0x6d, 0x6f, 0x64,
	0x65, 0x6c, 0x73, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x2e, 0x47, 0x65, 0x74, 0x42, 0x6f, 0x6f,
	0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x1f, 0x2e,
	0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x73, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x2e, 0x47, 0x65, 0x74,
	0x42, 0x6f, 0x6f, 0x74, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x52, 0x65, 0x70, 0x6c, 0x79, 0x22, 0x00,
	0x30, 0x01, 0x42, 0x1a, 0x5a, 0x18, 0x64, 0x69, 0x73, 0x74, 0x5f, 0x66, 0x69, 0x6e, 0x61, 0x6c,
	0x5f, 0x70, 0x72, 0x6f, 0x6a, 0x65, 0x63, 0x74, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x06,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_proto_modelservice_proto_rawDescOnce sync.Once
	file_proto_modelservice_proto_rawDescData = file_proto_modelservice_proto_rawDesc
)

func file_proto_modelservice_proto_rawDescGZIP() []byte {
	file_proto_modelservice_proto_rawDescOnce.Do(func() {
		file_proto_modelservice_proto_rawDescData = protoimpl.X.CompressGZIP(file_proto_modelservice_proto_rawDescData)
	})
	return file_proto_modelservice_proto_rawDescData
}

var file_proto_modelservice_proto_msgTypes = make([]protoimpl.MessageInfo, 10)
var file_proto_modelservice_proto_goTypes = []any{
	(*CollectModelsRequest)(nil),  // 0: modelservice.CollectModelsRequest
	(*CollectModelsResponse)(nil), // 1: modelservice.CollectModelsResponse
	(*GetModelRequest)(nil),       // 2: modelservice.GetModelRequest
	(*GetModelReply)(nil),         // 3: modelservice.GetModelReply
	(*GetBootModelRequest)(nil),   // 4: modelservice.GetBootModelRequest
	(*GetBootModelReply)(nil),     // 5: modelservice.GetBootModelReply
	(*GetPeerListRequest)(nil),    // 6: modelservice.GetPeerListRequest
	(*GetPeerListResponse)(nil),   // 7: modelservice.GetPeerListResponse
	(*Peer)(nil),                  // 8: modelservice.Peer
	nil,                           // 9: modelservice.GetPeerListResponse.PeersEntry
}
var file_proto_modelservice_proto_depIdxs = []int32{
	9, // 0: modelservice.GetPeerListResponse.peers:type_name -> modelservice.GetPeerListResponse.PeersEntry
	8, // 1: modelservice.GetPeerListResponse.PeersEntry.value:type_name -> modelservice.Peer
	2, // 2: modelservice.ModelService.GetModel:input_type -> modelservice.GetModelRequest
	6, // 3: modelservice.ModelService.GetPeerList:input_type -> modelservice.GetPeerListRequest
	0, // 4: modelservice.ModelService.CollectModels:input_type -> modelservice.CollectModelsRequest
	4, // 5: modelservice.ModelService.GetBootModel:input_type -> modelservice.GetBootModelRequest
	3, // 6: modelservice.ModelService.GetModel:output_type -> modelservice.GetModelReply
	7, // 7: modelservice.ModelService.GetPeerList:output_type -> modelservice.GetPeerListResponse
	1, // 8: modelservice.ModelService.CollectModels:output_type -> modelservice.CollectModelsResponse
	5, // 9: modelservice.ModelService.GetBootModel:output_type -> modelservice.GetBootModelReply
	6, // [6:10] is the sub-list for method output_type
	2, // [2:6] is the sub-list for method input_type
	2, // [2:2] is the sub-list for extension type_name
	2, // [2:2] is the sub-list for extension extendee
	0, // [0:2] is the sub-list for field type_name
}

func init() { file_proto_modelservice_proto_init() }
func file_proto_modelservice_proto_init() {
	if File_proto_modelservice_proto != nil {
		return
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_proto_modelservice_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   10,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_proto_modelservice_proto_goTypes,
		DependencyIndexes: file_proto_modelservice_proto_depIdxs,
		MessageInfos:      file_proto_modelservice_proto_msgTypes,
	}.Build()
	File_proto_modelservice_proto = out.File
	file_proto_modelservice_proto_rawDesc = nil
	file_proto_modelservice_proto_goTypes = nil
	file_proto_modelservice_proto_depIdxs = nil
}
